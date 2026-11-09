// A minimal single-client HTTP server and chat page for smolchat, written for Safari 12:
// the page is plain ES5 plus fetch and streams, with no newer CSS. It serves one request
// at a time, which is all the hardware can do anyway.
//
// POST /chat takes the message as a text body and streams the reply as chunked text. A
// 0x1e byte separates the reply from a final status line. Closing the connection stops
// generation. POST requests need an X-Smolchat header, so other websites open in Safari
// can't trigger them without a CORS preflight, which this server never approves.

#pragma once

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>

using piece_fn = std::function<bool(const char *, size_t)>;
using turn_fn  = std::function<bool(const std::string &, const piece_fn &, std::string &)>;

static const char WEB_PAGE[] = R"HTML(<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>smolchat</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; background: #f4f3ef; color: #24231f; font: 17px/1.45 -apple-system, Helvetica, Arial, sans-serif; }
  header { position: fixed; top: 0; left: 0; right: 0; height: 52px; padding: 0 16px; background: #fff; border-bottom: 1px solid #dedcd4; display: flex; align-items: center; justify-content: space-between; }
  h1 { margin: 0; font-size: 18px; font-weight: 600; }
  #log { max-width: 760px; margin: 0 auto; padding: 68px 16px 92px; }
  .msg { margin: 0 0 12px; padding: 10px 14px; border-radius: 14px; white-space: pre-wrap; word-wrap: break-word; }
  .user { margin-left: 15%; background: #2f5d50; color: #fff; }
  .bot { margin-right: 15%; background: #fff; border: 1px solid #dedcd4; }
  .bot.waiting { color: #827f74; }
  .stats { margin: -6px 0 16px 4px; color: #827f74; font-size: 13px; }
  footer { position: fixed; bottom: 0; left: 0; right: 0; padding: 12px 16px; background: #fff; border-top: 1px solid #dedcd4; }
  form { display: flex; max-width: 728px; margin: 0 auto; }
  input { flex: 1; min-width: 0; height: 44px; padding: 0 12px; font-size: 17px; border: 1px solid #c9c6bb; border-radius: 10px; -webkit-appearance: none; }
  button { height: 44px; padding: 0 18px; font-size: 16px; font-weight: 600; border: 0; border-radius: 10px; background: #2f5d50; color: #fff; -webkit-appearance: none; }
  form button { margin-left: 8px; }
  #reset { background: transparent; color: #2f5d50; padding: 0 4px; }
</style>
</head>
<body>
<header><h1>smolchat</h1><button id="reset" type="button">New chat</button></header>
<div id="log"></div>
<footer><form id="form"><input id="box" autocomplete="off" placeholder="Ask something" aria-label="Message"><button id="send" type="submit">Send</button></form></footer>
<script>
var log = document.getElementById('log');
var box = document.getElementById('box');
var send = document.getElementById('send');
var busy = false;
var reader = null;

function post(path, body) {
  return fetch(path, { method: 'POST', headers: { 'X-Smolchat': '1', 'Content-Type': 'text/plain' }, body: body || '' });
}
function bottom() { window.scrollTo(0, document.body.scrollHeight); }
function add(cls, text) {
  var d = document.createElement('div');
  d.className = cls;
  d.textContent = text;
  log.appendChild(d);
  bottom();
  return d;
}
function setBusy(b) { busy = b; send.textContent = b ? 'Stop' : 'Send'; }

function ask() {
  if (busy) { if (reader) { reader.cancel(); } return; }
  var text = box.value.replace(/^\s+|\s+$/g, '');
  if (!text) { return; }
  box.value = '';
  add('msg user', text);
  var m = add('msg bot waiting', 'Thinking…');
  var s = add('stats', '');
  var started = false;
  function show(t) {
    var i = t.indexOf('\x1e');
    var reply = i < 0 ? t : t.slice(0, i);
    if (reply) { started = true; m.className = 'msg bot'; m.textContent = reply; }
    if (i >= 0) { s.textContent = t.slice(i + 1); }
    bottom();
  }
  setBusy(true);
  post('/chat', text).then(function (r) {
    if (!r.ok) { throw new Error('the server answered ' + r.status); }
    if (!r.body) { return r.text().then(show); }
    reader = r.body.getReader();
    var dec = new TextDecoder();
    var all = '';
    function pump() {
      return reader.read().then(function (x) {
        if (x.done) { return; }
        all += dec.decode(x.value, { stream: true });
        show(all);
        return pump();
      });
    }
    return pump();
  }).catch(function (e) {
    s.textContent = 'Error: ' + e.message;
  }).then(function () {
    if (!started) { m.className = 'msg bot'; m.textContent = '(no reply)'; }
    reader = null;
    setBusy(false);
  });
}

document.getElementById('form').addEventListener('submit', function (e) { e.preventDefault(); ask(); });
document.getElementById('reset').addEventListener('click', function () {
  if (busy) { return; }
  post('/reset').then(function () { log.textContent = ''; box.focus(); });
});
post('/reset');
</script>
</body>
</html>
)HTML";

struct http_request {
    std::string method;
    std::string path;
    std::string body;
    bool has_token = false;
};

static bool send_all(int fd, const char * p, size_t n) {
    while (n > 0) {
        const ssize_t w = send(fd, p, n, 0);
        if (w < 0 && errno == EINTR) {
            continue;
        }
        if (w <= 0) {
            return false;
        }
        p += w;
        n -= (size_t) w;
    }
    return true;
}

static bool send_chunk(int fd, const char * p, size_t n) {
    char head[32];
    const int h = snprintf(head, sizeof(head), "%zx\r\n", n);
    return send_all(fd, head, h) && send_all(fd, p, n) && send_all(fd, "\r\n", 2);
}

static void respond(int fd, const char * status, const char * type, const char * body, size_t n) {
    char head[256];
    const int h = snprintf(head, sizeof(head),
        "HTTP/1.1 %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        status, type, n);
    if (send_all(fd, head, h)) {
        send_all(fd, body, n);
    }
}

static void respond_text(int fd, const char * status, const std::string & text) {
    respond(fd, status, "text/plain; charset=utf-8", text.c_str(), text.size());
}

static bool read_request(int fd, http_request & req) {
    const size_t max_head = 16 * 1024;
    const size_t max_body = 64 * 1024;

    std::string data;
    char buf[4096];
    size_t head_end;
    while ((head_end = data.find("\r\n\r\n")) == std::string::npos) {
        if (data.size() > max_head) {
            return false;
        }
        const ssize_t n = recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) {
            return false;
        }
        data.append(buf, n);
    }

    const size_t line_end = data.find("\r\n");
    const size_t sp1 = data.find(' ');
    const size_t sp2 = sp1 == std::string::npos ? sp1 : data.find(' ', sp1 + 1);
    if (sp2 == std::string::npos || sp2 > line_end) {
        return false;
    }
    req.method = data.substr(0, sp1);
    req.path = data.substr(sp1 + 1, sp2 - sp1 - 1);
    req.path = req.path.substr(0, req.path.find('?'));

    std::string head = data.substr(0, head_end + 2);
    std::transform(head.begin(), head.end(), head.begin(), [](unsigned char c) { return (char) tolower(c); });
    req.has_token = head.find("\r\nx-smolchat:") != std::string::npos;

    size_t length = 0;
    const size_t cl = head.find("\r\ncontent-length:");
    if (cl != std::string::npos) {
        const long v = strtol(head.c_str() + cl + 17, nullptr, 10);
        if (v < 0 || (size_t) v > max_body) {
            return false;
        }
        length = (size_t) v;
    }

    req.body = data.substr(head_end + 4);
    while (req.body.size() < length) {
        const ssize_t n = recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) {
            return false;
        }
        req.body.append(buf, n);
    }
    req.body.resize(length);
    return true;
}

static void handle_connection(int fd, const turn_fn & turn, const std::function<void()> & reset) {
    const timeval timeout = { 15, 0 };
    const int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)); // send each token immediately

    http_request req;
    if (!read_request(fd, req)) {
        return;
    }

    if (req.method == "GET" && req.path == "/") {
        respond(fd, "200 OK", "text/html; charset=utf-8", WEB_PAGE, sizeof(WEB_PAGE) - 1);
        return;
    }
    if (req.method != "POST" || (req.path != "/chat" && req.path != "/reset")) {
        respond_text(fd, "404 Not Found", "not found\n");
        return;
    }
    if (!req.has_token) {
        respond_text(fd, "403 Forbidden", "missing X-Smolchat header\n");
        return;
    }
    if (req.path == "/reset") {
        reset();
        respond(fd, "204 No Content", "text/plain", "", 0);
        return;
    }
    if (req.body.empty()) {
        respond_text(fd, "400 Bad Request", "empty message\n");
        return;
    }

    static const char head[] =
        "HTTP/1.1 200 OK\r\nContent-Type: text/plain; charset=utf-8\r\n"
        "Transfer-Encoding: chunked\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n";
    if (!send_all(fd, head, sizeof(head) - 1)) {
        return;
    }

    std::string stats;
    const bool ok = turn(req.body, [fd](const char * p, size_t n) { return send_chunk(fd, p, n); }, stats);
    const std::string tail = std::string("\x1e") + (ok ? "" : "Error: ") + stats;
    if (send_chunk(fd, tail.c_str(), tail.size())) {
        send_all(fd, "0\r\n\r\n", 5);
    }
}

static int serve_http(const std::string & host, int port, const turn_fn & turn, const std::function<void()> & reset) {
    signal(SIGPIPE, SIG_IGN); // a closed page must not kill the server

    const int srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) {
        perror("socket");
        return 1;
    }
    const int one = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t) port);
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        fprintf(stderr, "error: --host must be an IPv4 address\n");
        return 1;
    }
    if (bind(srv, (sockaddr *) &addr, sizeof(addr)) != 0 || listen(srv, 8) != 0) {
        perror("bind");
        return 1;
    }

    printf("smolchat serving on http://%s:%d/\n", host.c_str(), port);
    fflush(stdout);

    while (true) {
        const int fd = accept(srv, nullptr, nullptr);
        if (fd < 0) {
            if (errno == EINTR || errno == ECONNABORTED) {
                continue;
            }
            perror("accept");
            return 1;
        }
        handle_connection(fd, turn, reset);
        close(fd);
    }
}
