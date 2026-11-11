// tinyppl: perplexity on the iPad with llama.h only (llama.cpp's common/ needs std::filesystem, which iOS 12 lacks).
// Scores like llama-perplexity: the text is cut into chunks of n_ctx tokens and the second half of each chunk is scored, but every
// token is decoded on its own (like generation, so cache-aware routing sees the same pool states as in chat). With a pruned LM head
// the dropped vocabulary has logit -inf; a target outside the head is counted as a miss and left out of the average.
// usage: tinyppl -m model.gguf -f text.txt [-c 512] [--chunks N] [-t 2] [--experts K]
#include "llama.h"
#include "gguf.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    std::string model_path, text_path; int n_ctx = 512, n_chunks = 2, n_threads = 2, n_experts = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i]; const char * v = argv[i + 1];
        if (a == "-m") model_path = v; else if (a == "-f") text_path = v; else if (a == "-c") n_ctx = atoi(v);
        else if (a == "--chunks") n_chunks = atoi(v); else if (a == "-t") n_threads = atoi(v); else if (a == "--experts") n_experts = atoi(v);
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 1; }
    }
    FILE * f = fopen(text_path.c_str(), "rb"); if (!f) { fprintf(stderr, "cannot read %s\n", text_path.c_str()); return 1; }
    std::string text; char buf[65536]; size_t r; while ((r = fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, r); fclose(f);

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    std::vector<llama_model_kv_override> kvo;
    if (n_experts > 0) {
        gguf_init_params gp = { true, nullptr }; gguf_context * g = gguf_init_from_file(model_path.c_str(), gp);
        std::string arch = "llama";
        if (g) { int64_t k = gguf_find_key(g, "general.architecture"); if (k >= 0) arch = gguf_get_val_str(g, k); gguf_free(g); }
        llama_model_kv_override ov = {}; ov.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
        snprintf(ov.key, sizeof ov.key, "%s.expert_used_count", arch.c_str()); ov.val_i64 = n_experts;
        kvo.push_back(ov); kvo.push_back({}); mp.kv_overrides = kvo.data();
        setenv("GGML_MOE_USED", std::to_string(n_experts).c_str(), 0);
    }
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "cannot load %s\n", model_path.c_str()); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx; cp.n_batch = n_ctx; cp.n_ubatch = 1; cp.n_threads = n_threads; cp.n_threads_batch = n_threads;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "cannot create context\n"); return 1; }

    const int nt = -llama_tokenize(vocab, text.c_str(), (int) text.size(), nullptr, 0, false, false);
    std::vector<llama_token> toks(nt);
    llama_tokenize(vocab, text.c_str(), (int) text.size(), toks.data(), nt, false, false);
    const int max_chunks = nt / n_ctx; if (n_chunks > max_chunks) n_chunks = max_chunks;
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    fprintf(stderr, "tinyppl: %d tokens, %d chunks of %d, bos %d\n", nt, n_chunks, n_ctx, (int) add_bos);

    llama_batch_ext * batch = llama_batch_ext_init(ctx);
    llama_memory_t mem = llama_get_memory(ctx);
    double nll = 0.0; long scored = 0, missed = 0;
    for (int c = 0; c < n_chunks; c++) {
        llama_memory_clear(mem, true);
        std::vector<llama_token> ch(toks.begin() + (size_t) c * n_ctx, toks.begin() + (size_t) (c + 1) * n_ctx);
        if (add_bos) ch[0] = llama_vocab_bos(vocab);
        for (int i = 0; i < n_ctx - 1; i++) {
            llama_batch_ext_clear(batch);
            const int32_t idx = llama_batch_ext_add_token(batch, 0, ch[i]);
            const llama_pos pos = i; llama_batch_ext_set_pos(batch, idx, &pos);
            llama_batch_ext_set_output_logits(batch, 0, true);
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
            if (i < n_ctx / 2) continue;
            const float * lg = llama_get_logits_ith(ctx, -1);
            const float t = lg[ch[i + 1]];
            if (!std::isfinite(t)) { missed++; continue; }
            float mx = -INFINITY; for (int v = 0; v < n_vocab; v++) if (lg[v] > mx) mx = lg[v];
            double s = 0.0; for (int v = 0; v < n_vocab; v++) if (std::isfinite(lg[v])) s += std::exp((double) lg[v] - mx);
            nll += -((double) t - mx - std::log(s)); scored++;
        }
        fprintf(stderr, "[%d] running PPL %.4f (%ld scored, %ld outside the head)\n", c + 1, std::exp(nll / scored), scored, missed);
    }
    printf("PPL = %.4f over %ld tokens (%ld targets outside the head, %.2f %%)\n", std::exp(nll / scored), scored, missed,
           100.0 * missed / (scored + missed));
    llama_batch_ext_free(batch); llama_free(ctx); llama_model_free(model);
    return 0;
}
