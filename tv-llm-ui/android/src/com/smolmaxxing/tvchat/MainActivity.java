package com.smolmaxxing.tvchat;

import android.app.Activity;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;

import java.net.HttpURLConnection;
import java.net.URL;

/** Shows the chat page from the local server in a WebView, after a "loading" screen while the model loads. */
public class MainActivity extends Activity {
    private static final String PAGE = "http://127.0.0.1:" + ChatService.PORT + "/";
    private WebView web;
    private final Handler main = new Handler(Looper.getMainLooper());
    private volatile boolean polling;

    @Override
    protected void onCreate(Bundle saved) {
        super.onCreate(saved);
        ChatService.start(this);   // normally already running since boot
        web = new WebView(this);
        WebSettings s = web.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);
        web.setWebViewClient(new WebViewClient());
        web.setBackgroundColor(0xff111111);
        setContentView(web);
        status("Starting the model…");
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (web.getUrl() == null || !web.getUrl().startsWith("http")) waitForServer();
    }

    @Override
    protected void onPause() {
        polling = false;
        super.onPause();
    }

    private void waitForServer() {
        polling = true;
        new Thread(() -> {
            int tries = 0;
            while (polling) {
                if (healthy()) {
                    main.post(() -> web.loadUrl(PAGE));
                    return;
                }
                if (++tries == 30 && !ChatService.modelFile(this).canRead()) {
                    main.post(() -> status("Model file missing:<br><code>" + ChatService.modelFile(this)
                            + "</code><br><br>Install it from the Mac with <code>tv-llm-ui/android/install.sh</code>."));
                }
                try { Thread.sleep(1000); } catch (InterruptedException e) { return; }
            }
        }, "smolchat-health").start();
    }

    private static boolean healthy() {
        try {
            HttpURLConnection c = (HttpURLConnection) new URL(PAGE + "health").openConnection();
            c.setConnectTimeout(1000);
            c.setReadTimeout(1000);
            boolean ok = c.getResponseCode() == 200;
            c.disconnect();
            return ok;
        } catch (Exception e) {
            return false;
        }
    }

    private void status(String html) {
        web.loadDataWithBaseURL(null,
                "<body style='background:#111;color:#ddd;font:28px sans-serif;display:flex;align-items:center;"
                        + "justify-content:center;height:90vh;text-align:center'><div>" + html + "</div></body>",
                "text/html", "utf-8", null);
    }
}
