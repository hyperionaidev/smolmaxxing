// smolchat: a small interactive chat on top of the llama.cpp C API, for devices with
// very little RAM. Based on llama.cpp's examples/simple-chat, plus: a repeat penalty,
// a thread count that defaults to the iPad Air's two cores, recovery when the context
// fills up, and a speed line after every reply.
//
// The whole history is re-rendered every turn and compared with what the model has
// already evaluated, so only the new tokens are decoded. When the history no longer
// fits, the oldest turns are dropped.

#include "llama.h"

#include <unistd.h>

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

struct options {
    std::string model;
    std::string system_prompt;
    int      n_threads      = 2;
    int      n_ctx          = 1024;
    int      n_predict      = 400;
    int      repeat_last    = 256;   // repeat penalty settings from smolmaxxing EXPERIMENT.md
    float    repeat_penalty = 1.15f;
    float    temp           = 0.7f;
    uint32_t seed           = LLAMA_DEFAULT_SEED;
    bool     no_think       = false;
};

struct message {
    std::string role;
    std::string content;
};

static const char * const EMPTY_THINK = "<think>\n\n</think>\n\n";

static void usage(const char * name) {
    printf("usage: %s -m model.gguf [options]\n"
           "  -t N                threads (default 2)\n"
           "  -c N                context size in tokens (default 1024)\n"
           "  -n N                maximum tokens per reply (default 400)\n"
           "  -s TEXT             system prompt\n"
           "  --temp X            temperature (default 0.7)\n"
           "  --repeat-penalty X  repeat penalty, 1.0 turns it off (default 1.15)\n"
           "  --repeat-last N     tokens the penalty looks back over (default 256)\n"
           "  --seed N            sampling seed\n"
           "  --no-think          skip the reasoning block on models that have one\n"
           "\nIn the chat: /reset clears the conversation, /quit exits.\n", name);
}

static bool parse_args(int argc, char ** argv, options & o) {
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--no-think") {
            o.no_think = true;
            continue;
        }
        if (i + 1 >= argc) {
            return false;
        }
        const char * v = argv[++i];
        if      (a == "-m")               { o.model = v; }
        else if (a == "-s")               { o.system_prompt = v; }
        else if (a == "-t")               { o.n_threads = atoi(v); }
        else if (a == "-c")               { o.n_ctx = atoi(v); }
        else if (a == "-n")               { o.n_predict = atoi(v); }
        else if (a == "--temp")           { o.temp = (float) atof(v); }
        else if (a == "--repeat-penalty") { o.repeat_penalty = (float) atof(v); }
        else if (a == "--repeat-last")    { o.repeat_last = atoi(v); }
        else if (a == "--seed")           { o.seed = (uint32_t) strtoul(v, nullptr, 10); }
        else                              { return false; }
    }
    return !o.model.empty() && o.n_threads > 0 && o.n_ctx > 0 && o.n_predict > 0;
}

// returns false if the model's chat template is not one llama.cpp can apply
static bool render(const char * tmpl, const std::vector<message> & history, std::string & out) {
    std::vector<llama_chat_message> msgs;
    msgs.reserve(history.size());
    for (const auto & m : history) {
        msgs.push_back({ m.role.c_str(), m.content.c_str() });
    }
    std::vector<char> buf(4096);
    int n = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), true, buf.data(), buf.size());
    if (n > (int) buf.size()) {
        buf.resize(n);
        n = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), true, buf.data(), buf.size());
    }
    if (n < 0) {
        return false;
    }
    out.assign(buf.data(), n);
    return true;
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    const int n = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, true, true);
    std::vector<llama_token> tokens(n);
    if (llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), true, true) < 0) {
        tokens.clear();
    }
    return tokens;
}

// evaluate tokens at consecutive positions from pos_0, keeping logits for the last one only
static bool decode(llama_context * ctx, llama_batch_ext * batch, const llama_token * tokens, int n, llama_pos pos_0) {
    llama_batch_ext_clear(batch);
    for (int i = 0; i < n; i++) {
        const int32_t idx = llama_batch_ext_add_token(batch, 0, tokens[i]);
        const llama_pos pos = pos_0 + i;
        llama_batch_ext_set_pos(batch, idx, &pos);
    }
    llama_batch_ext_set_output_logits(batch, n - 1, true);
    return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) == 0;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    options o;
    if (!parse_args(argc, argv, o)) {
        usage(argv[0]);
        return 1;
    }

    const bool tty = isatty(STDOUT_FILENO);
    const char * c_user  = tty ? "\033[32m" : "";
    const char * c_model = tty ? "\033[33m" : "";
    const char * c_stats = tty ? "\033[90m" : "";
    const char * c_reset = tty ? "\033[0m"  : "";

    llama_log_set([](enum ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_ERROR) {
            fprintf(stderr, "%s", text);
        }
    }, nullptr);

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(o.model.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "error: unable to load %s\n", o.model.c_str());
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const char * tmpl = llama_model_chat_template(model, nullptr);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx           = o.n_ctx;
    ctx_params.n_batch         = o.n_ctx;
    ctx_params.n_threads       = o.n_threads;
    ctx_params.n_threads_batch = o.n_threads;
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "error: failed to create the context\n");
        return 1;
    }
    const int n_ctx = (int) llama_n_ctx(ctx);

    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(llama_vocab_n_tokens(vocab), o.repeat_last, o.repeat_penalty, 0.0f, 0.0f));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(o.temp));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(o.seed));

    llama_batch_ext * batch = llama_batch_ext_init(ctx);
    llama_memory_t mem = llama_get_memory(ctx);

    std::vector<message> history;
    if (!o.system_prompt.empty()) {
        history.push_back({ "system", o.system_prompt });
    }
    const size_t first_turn = history.size();

    std::vector<llama_token> evaluated; // exactly what the model's memory holds
    const int reserve = std::min(o.n_predict, n_ctx / 4); // room kept free for the reply

    printf("smolchat ready. /reset clears the conversation, /quit exits.\n");

    while (true) {
        printf("%s> %s", c_user, c_reset);
        fflush(stdout);

        std::string user;
        if (!std::getline(std::cin, user) || user == "/quit" || user == "/exit") {
            break;
        }
        if (user.empty()) {
            continue;
        }
        if (user == "/reset") {
            history.resize(first_turn);
            evaluated.clear();
            llama_memory_clear(mem, true);
            printf("%s[conversation cleared]%s\n", c_stats, c_reset);
            continue;
        }

        history.push_back({ "user", user });

        // render the conversation, dropping the oldest turns until it fits
        std::string prompt;
        std::vector<llama_token> tokens;
        bool ok = true;
        size_t dropped = 0;
        while (true) {
            if (!render(tmpl, history, prompt)) {
                fprintf(stderr, "error: this model's chat template is not supported\n");
                ok = false;
                break;
            }
            if (o.no_think) {
                prompt += EMPTY_THINK;
            }
            tokens = tokenize(vocab, prompt);
            if (tokens.empty()) {
                fprintf(stderr, "error: failed to tokenize the conversation\n");
                ok = false;
                break;
            }
            if ((int) tokens.size() + reserve <= n_ctx) {
                break;
            }
            if (history.size() <= first_turn + 1) {
                fprintf(stderr, "error: that message is too long for a context of %d tokens\n", n_ctx);
                ok = false;
                break;
            }
            history.erase(history.begin() + first_turn, history.begin() + first_turn + 2);
            dropped += 2;
        }
        if (!ok) {
            history.pop_back();
            continue;
        }
        if (dropped > 0) {
            printf("%s[context full: dropped the %zu oldest messages]%s\n", c_stats, dropped, c_reset);
        }

        // keep whatever is already evaluated, always re-evaluating at least one token
        size_t keep = 0;
        while (keep < evaluated.size() && keep < tokens.size() && evaluated[keep] == tokens[keep]) {
            keep++;
        }
        if (keep == tokens.size()) {
            keep--;
        }
        if (keep < evaluated.size() && !llama_memory_seq_rm(mem, 0, keep, -1)) {
            // some architectures keep recurrent state and cannot rewind partway
            llama_memory_clear(mem, true);
            keep = 0;
        }

        const int64_t t_start = llama_time_us();
        if (!decode(ctx, batch, tokens.data() + keep, (int) (tokens.size() - keep), (llama_pos) keep)) {
            fprintf(stderr, "error: failed to evaluate the prompt\n");
            history.pop_back();
            evaluated.clear();
            llama_memory_clear(mem, true);
            continue;
        }
        const int64_t t_prompt = llama_time_us();
        const size_t n_prompt = tokens.size() - keep;
        evaluated = tokens;

        printf("%s", c_model);
        llama_sampler_reset(smpl);

        std::string response;
        int n_generated = 0;
        const char * stop = "";
        while (true) {
            const llama_token id = llama_sampler_sample(smpl, ctx, -1);
            if (llama_vocab_is_eog(vocab, id)) {
                break;
            }
            char buf[256];
            const int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, false);
            if (n > 0) {
                fwrite(buf, 1, n, stdout);
                fflush(stdout);
                response.append(buf, n);
            }
            n_generated++;

            if (n_generated >= o.n_predict) {
                stop = ", stopped at the reply limit";
                break;
            }
            if ((int) evaluated.size() + 1 >= n_ctx) {
                stop = ", stopped because the context is full";
                break;
            }
            if (!decode(ctx, batch, &id, 1, (llama_pos) evaluated.size())) {
                stop = ", stopped by a decode error";
                break;
            }
            evaluated.push_back(id);
        }
        const int64_t t_end = llama_time_us();

        const double s_prompt = (t_prompt - t_start) / 1e6;
        const double s_gen    = (t_end - t_prompt) / 1e6;
        printf("%s\n%s[%d tok, %.2f tok/s; prompt %zu tok, %.2f tok/s%s]%s\n", c_reset, c_stats,
               n_generated, s_gen > 0 ? n_generated / s_gen : 0.0,
               n_prompt, s_prompt > 0 ? n_prompt / s_prompt : 0.0, stop, c_reset);

        // keep the empty reasoning block in the history so the next turn's tokens line up
        history.push_back({ "assistant", (o.no_think ? std::string(EMPTY_THINK) : std::string()) + response });
    }

    llama_batch_ext_free(batch);
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
