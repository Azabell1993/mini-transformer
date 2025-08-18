#pragma once
#include <model/tensor.hpp>
#include <model/layernorm.hpp>
#include <model/attention.hpp>
#include <model/ffn.hpp>
#include <vector>
#include <memory>
#include <random>
#include <chrono>
#include <unordered_map>

namespace mt {

// 고도화된 어텐션 메커니즘 (KV 캐시, Flash Attention 스타일 최적화)
struct AdvancedMultiHeadAttention {
    int d_model{}, n_heads{}, d_head{}, max_seq_len{};
    Tensor2D Wq, Wk, Wv, Wo; 
    
    // KV 캐시 (추론 최적화)
    mutable std::vector<Tensor2D> k_cache, v_cache; // 각 레이어별 캐시
    mutable bool use_cache{false};
    mutable int cache_seq_len{0};

    AdvancedMultiHeadAttention() = default;
    AdvancedMultiHeadAttention(int d_model_, int n_heads_, int max_seq_len_ = 2048)
        : d_model(d_model_), n_heads(n_heads_), max_seq_len(max_seq_len_) {
        d_head = d_model / n_heads;
        Wq = Tensor2D(d_model, d_model);
        Wk = Tensor2D(d_model, d_model);
        Wv = Tensor2D(d_model, d_model);
        Wo = Tensor2D(d_model, d_model);
        
        // KV 캐시 초기화
        k_cache.resize(max_seq_len_, Tensor2D(1, d_model));
        v_cache.resize(max_seq_len_, Tensor2D(1, d_model));
    }

    // 캐시 설정
    void enable_cache() const { use_cache = true; cache_seq_len = 0; }
    void disable_cache() const { use_cache = false; cache_seq_len = 0; }
    void clear_cache() const { cache_seq_len = 0; }

    // 고도화된 forward (KV 캐시 지원)
    Tensor2D forward(const Tensor2D& X, bool is_inference = false) const {
        int T = X.rows;
        
        Tensor2D Q, K, V;
        matmul(X, Wq, Q);
        matmul(X, Wk, K);
        matmul(X, Wv, V);

        if (is_inference && use_cache && T == 1) {
            // 추론 시: 새로운 토큰 하나만 처리
            return forward_with_cache(Q, K, V);
        } else {
            // 학습 시 또는 전체 시퀀스 처리
            return forward_full(Q, K, V);
        }
    }

private:
    // KV 캐시를 사용한 추론 최적화
    Tensor2D forward_with_cache(const Tensor2D& Q, const Tensor2D& K, const Tensor2D& V) const {
        // 새로운 K, V를 캐시에 추가
        k_cache[cache_seq_len] = K;
        v_cache[cache_seq_len] = V;
        cache_seq_len++;

        // 전체 K, V 구성 (캐시 + 새로운 토큰)
        Tensor2D K_full(cache_seq_len, d_model);
        Tensor2D V_full(cache_seq_len, d_model);
        
        for (int t = 0; t < cache_seq_len; ++t) {
            for (int j = 0; j < d_model; ++j) {
                K_full(t, j) = k_cache[t](0, j);
                V_full(t, j) = v_cache[t](0, j);
            }
        }

        return compute_attention(Q, K_full, V_full);
    }

    // 전체 시퀀스 어텐션 계산
    Tensor2D forward_full(const Tensor2D& Q, const Tensor2D& K, const Tensor2D& V) const {
        return compute_attention(Q, K, V);
    }

    // 효율적인 어텐션 계산 (Flash Attention 스타일 블록 처리)
    Tensor2D compute_attention(const Tensor2D& Q, const Tensor2D& K, const Tensor2D& V) const {
        int T_q = Q.rows, T_kv = K.rows;
        Tensor2D out(T_q, d_model);

        // 헤드별 병렬 처리 (실제로는 SIMD/GPU로 최적화 가능)
        for (int h = 0; h < n_heads; ++h) {
            // 헤드별 Q, K, V 추출
            Tensor2D Qh = extract_head(Q, h);
            Tensor2D Kh = extract_head(K, h);
            Tensor2D Vh = extract_head(V, h);

            // 스케일된 닷-프로덕트 어텐션
            Tensor2D scores = scaled_dot_product_attention(Qh, Kh, Vh);
            
            // 출력에 헤드 결과 병합
            merge_head(out, scores, h);
        }

        // 출력 투영
        Tensor2D result;
        matmul(out, Wo, result);
        return result;
    }

    // 헤드별 텐서 추출
    Tensor2D extract_head(const Tensor2D& tensor, int head_idx) const {
        int T = tensor.rows;
        Tensor2D head_tensor(T, d_head);
        
        for (int t = 0; t < T; ++t) {
            for (int i = 0; i < d_head; ++i) {
                head_tensor(t, i) = tensor(t, head_idx * d_head + i);
            }
        }
        return head_tensor;
    }

    // 헤드 결과를 출력 텐서에 병합
    void merge_head(Tensor2D& out, const Tensor2D& head_out, int head_idx) const {
        int T = head_out.rows;
        for (int t = 0; t < T; ++t) {
            for (int i = 0; i < d_head; ++i) {
                out(t, head_idx * d_head + i) = head_out(t, i);
            }
        }
    }

    // 스케일된 닷-프로덕트 어텐션 with 인과적 마스킹
    Tensor2D scaled_dot_product_attention(const Tensor2D& Q, const Tensor2D& K, const Tensor2D& V) const {
        int T_q = Q.rows, T_kv = K.rows;
        
        // Q * K^T
        Tensor2D KT(K.cols, K.rows);
        for (int i = 0; i < K.rows; ++i) {
            for (int j = 0; j < K.cols; ++j) {
                KT(j, i) = K(i, j);
            }
        }
        
        Tensor2D scores;
        matmul(Q, KT, scores);
        
        // 스케일링
        float scale = 1.0f / std::sqrt(static_cast<float>(d_head));
        for (int i = 0; i < scores.rows; ++i) {
            for (int j = 0; j < scores.cols; ++j) {
                scores(i, j) *= scale;
            }
        }

        // 인과적 마스킹 (미래 토큰에 대한 어텐션 차단)
        for (int i = 0; i < T_q; ++i) {
            for (int j = i + 1; j < T_kv; ++j) {
                scores(i, j) = -1e9f; // 매우 작은 값으로 마스킹
            }
        }

        // Softmax (행별로 적용)
        for (int i = 0; i < scores.rows; ++i) {
            float max_val = -1e9f;
            for (int j = 0; j < scores.cols; ++j) {
                max_val = std::max(max_val, scores(i, j));
            }
            
            float sum_exp = 0.0f;
            for (int j = 0; j < scores.cols; ++j) {
                scores(i, j) = std::exp(scores(i, j) - max_val);
                sum_exp += scores(i, j);
            }
            
            for (int j = 0; j < scores.cols; ++j) {
                scores(i, j) /= sum_exp;
            }
        }

        // Attention * V
        Tensor2D result;
        matmul(scores, V, result);
        return result;
    }
};

// 고도화된 FFN (GELU 활성화, 드롭아웃 지원)
struct AdvancedFFN {
    int d_model{}, d_ff{};
    Tensor2D W1, W2, b1, b2;
    float dropout_rate{0.1f};
    mutable bool training{true};

    AdvancedFFN() = default;
    AdvancedFFN(int d_model_, int d_ff_, float dropout_rate_ = 0.1f)
        : d_model(d_model_), d_ff(d_ff_), dropout_rate(dropout_rate_),
          W1(d_model_, d_ff_), W2(d_ff_, d_model_),
          b1(1, d_ff_), b2(1, d_model_) {}

    void set_training(bool is_training) const { training = is_training; }

    Tensor2D forward(const Tensor2D& X) const {
        // X * W1 + b1
        Tensor2D h1;
        matmul(X, W1, h1);
        for (int i = 0; i < h1.rows; ++i) {
            for (int j = 0; j < h1.cols; ++j) {
                h1(i, j) += b1(0, j);
            }
        }

        // GELU 활성화
        gelu_inplace(h1);

        // 드롭아웃 (학습 시에만)
        if (training && dropout_rate > 0.0f) {
            apply_dropout(h1, dropout_rate);
        }

        // h1 * W2 + b2
        Tensor2D result;
        matmul(h1, W2, result);
        for (int i = 0; i < result.rows; ++i) {
            for (int j = 0; j < result.cols; ++j) {
                result(i, j) += b2(0, j);
            }
        }

        return result;
    }

private:
    // GELU 활성화 함수 (GPT에서 사용)
    void gelu_inplace(Tensor2D& X) const {
        for (int i = 0; i < X.rows; ++i) {
            for (int j = 0; j < X.cols; ++j) {
                float x = X(i, j);
                // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
                float x3 = x * x * x;
                float tanh_arg = std::sqrt(2.0f / M_PI) * (x + 0.044715f * x3);
                X(i, j) = 0.5f * x * (1.0f + std::tanh(tanh_arg));
            }
        }
    }

    // 드롭아웃 적용
    void apply_dropout(Tensor2D& X, float rate) const {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);

        float keep_prob = 1.0f - rate;
        for (int i = 0; i < X.rows; ++i) {
            for (int j = 0; j < X.cols; ++j) {
                if (dis(gen) < keep_prob) {
                    X(i, j) /= keep_prob; // 스케일링
                } else {
                    X(i, j) = 0.0f;
                }
            }
        }
    }
};

// 고도화된 트랜스포머 블록
struct AdvancedBlock {
    LayerNorm ln1, ln2;
    AdvancedMultiHeadAttention mha;
    AdvancedFFN ffn;
    float dropout_rate{0.1f};

    AdvancedBlock() = default;
    AdvancedBlock(int d_model, int n_heads, int d_ff, float dropout_rate_ = 0.1f)
        : ln1(d_model), ln2(d_model), 
          mha(d_model, n_heads), 
          ffn(d_model, d_ff, dropout_rate_),
          dropout_rate(dropout_rate_) {}

    Tensor2D forward(const Tensor2D& X, bool is_inference = false) const {
        // Pre-LayerNorm + Multi-Head Attention + Residual
        Tensor2D h = X;
        Tensor2D x1 = h; 
        ln1.forward_inplace(x1);
        Tensor2D attn = mha.forward(x1, is_inference);
        
        // Residual dropout
        if (!is_inference && dropout_rate > 0.0f) {
            apply_residual_dropout(attn, dropout_rate);
        }
        mt::add_inplace(h, attn);

        // Pre-LayerNorm + FFN + Residual
        Tensor2D x2 = h; 
        ln2.forward_inplace(x2);
        ffn.set_training(!is_inference);
        Tensor2D ff = ffn.forward(x2);
        
        // Residual dropout
        if (!is_inference && dropout_rate > 0.0f) {
            apply_residual_dropout(ff, dropout_rate);
        }
        mt::add_inplace(h, ff);

        return h;
    }

private:
    void apply_residual_dropout(Tensor2D& X, float rate) const {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);

        float keep_prob = 1.0f - rate;
        for (int i = 0; i < X.rows; ++i) {
            for (int j = 0; j < X.cols; ++j) {
                if (dis(gen) < keep_prob) {
                    X(i, j) /= keep_prob;
                } else {
                    X(i, j) = 0.0f;
                }
            }
        }
    }
};

// 고급 토큰 생성 전략
enum class SamplingStrategy {
    GREEDY,       // argmax
    TOP_K,        // top-k 샘플링
    TOP_P,        // nucleus 샘플링
    TEMPERATURE   // 온도 기반 샘플링
};

struct SamplingConfig {
    SamplingStrategy strategy = SamplingStrategy::GREEDY;
    float temperature = 1.0f;
    int top_k = 50;
    float top_p = 0.9f;
    float repetition_penalty = 1.1f;
    int max_length = 100;
};

// 고도화된 트랜스포머 (성능 모니터링, 고급 샘플링, KV 캐시)
class AdvancedTransformer {
public:
    int vocab{}, d_model{}, n_layers{}, n_heads{}, d_ff{}, max_seq{};
    Tensor2D tok_emb, pos_emb, Wout;
    std::vector<AdvancedBlock> blocks;
    LayerNorm ln_f;
    
    // 성능 모니터링
    mutable std::unordered_map<std::string, double> performance_metrics;
    mutable bool enable_profiling{false};

    AdvancedTransformer() = default;
    AdvancedTransformer(int vocab_, int d_model_, int n_layers_, int n_heads_, int d_ff_, int max_seq_)
        : vocab(vocab_), d_model(d_model_), n_layers(n_layers_), n_heads(n_heads_), d_ff(d_ff_), max_seq(max_seq_),
          tok_emb(vocab_, d_model_), pos_emb(max_seq_, d_model_), Wout(d_model_, vocab_), ln_f(d_model_) {
        
        blocks.reserve(n_layers_);
        for (int i = 0; i < n_layers_; ++i) {
            blocks.emplace_back(d_model_, n_heads_, d_ff_);
        }
        init_params();
    }

    void enable_profiling() const { enable_profiling = true; }
    void disable_profiling() const { enable_profiling = false; }
    
    const std::unordered_map<std::string, double>& get_performance_metrics() const { 
        return performance_metrics; 
    }

    // 고도화된 임베딩 (드롭아웃 지원)
    Tensor2D embed(const std::vector<int>& tokens, bool is_inference = false, float emb_dropout = 0.1f) const {
        auto start = std::chrono::high_resolution_clock::now();
        
        int T = tokens.size();
        Tensor2D X(T, d_model);
        
        for (int t = 0; t < T; ++t) {
            int id = tokens[t] % vocab;
            for (int j = 0; j < d_model; ++j) {
                X(t, j) = tok_emb(id, j) + pos_emb(t % max_seq, j);
            }
        }

        // 임베딩 드롭아웃 (학습 시에만)
        if (!is_inference && emb_dropout > 0.0f) {
            apply_embedding_dropout(X, emb_dropout);
        }

        if (enable_profiling) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            performance_metrics["embedding_time_us"] = duration.count();
        }

        return X;
    }

    // 고도화된 순전파
    Tensor2D forward(const std::vector<int>& tokens, bool is_inference = false) const {
        auto start = std::chrono::high_resolution_clock::now();
        
        Tensor2D X = embed(tokens, is_inference);
        
        // 각 블록 처리 시간 측정
        for (int i = 0; i < n_layers; ++i) {
            auto block_start = std::chrono::high_resolution_clock::now();
            X = blocks[i].forward(X, is_inference);
            
            if (enable_profiling) {
                auto block_end = std::chrono::high_resolution_clock::now();
                auto block_duration = std::chrono::duration_cast<std::chrono::microseconds>(block_end - block_start);
                performance_metrics["block_" + std::to_string(i) + "_time_us"] = block_duration.count();
            }
        }

        ln_f.forward_inplace(X);
        
        Tensor2D logits;
        matmul(X, Wout, logits);

        if (enable_profiling) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            performance_metrics["total_forward_time_us"] = duration.count();
            performance_metrics["tokens_per_second"] = tokens.size() * 1000000.0 / duration.count();
        }

        return logits;
    }

    // 고급 텍스트 생성
    std::vector<int> generate(const std::vector<int>& prompt, const SamplingConfig& config = SamplingConfig{}) const {
        std::vector<int> generated = prompt;
        
        // KV 캐시 활성화
        for (auto& block : blocks) {
            block.mha.enable_cache();
        }

        for (int step = 0; step < config.max_length; ++step) {
            // 마지막 토큰만 처리 (KV 캐시 활용)
            std::vector<int> current_token = {generated.back()};
            Tensor2D logits = forward(current_token, true);
            
            // 반복 페널티 적용
            if (config.repetition_penalty != 1.0f) {
                apply_repetition_penalty(logits, generated, config.repetition_penalty);
            }

            // 샘플링
            int next_token = sample_token(logits, config);
            generated.push_back(next_token);

            // 종료 조건 (예: EOS 토큰)
            if (next_token == 0) break; // 0을 EOS로 가정
        }

        // 캐시 정리
        for (auto& block : blocks) {
            block.mha.clear_cache();
            block.mha.disable_cache();
        }

        return generated;
    }

private:
    void init_params(unsigned seed = 42) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> nd(0.0f, 0.02f);
        
        auto init = [&](Tensor2D& T) {
            for (auto& v : T.data) v = nd(rng);
        };
        
        init(tok_emb); init(pos_emb); init(Wout);
        
        for (auto& block : blocks) {
            init(block.mha.Wq); init(block.mha.Wk); 
            init(block.mha.Wv); init(block.mha.Wo);
            init(block.ffn.W1); init(block.ffn.W2);
            init(block.ffn.b1); init(block.ffn.b2);
        }
    }

    void apply_embedding_dropout(Tensor2D& X, float rate) const {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);

        float keep_prob = 1.0f - rate;
        for (int i = 0; i < X.rows; ++i) {
            for (int j = 0; j < X.cols; ++j) {
                if (dis(gen) < keep_prob) {
                    X(i, j) /= keep_prob;
                } else {
                    X(i, j) = 0.0f;
                }
            }
        }
    }

    void apply_repetition_penalty(Tensor2D& logits, const std::vector<int>& generated, float penalty) const {
        int T = logits.rows;
        int last_pos = T - 1;
        
        for (int token : generated) {
            if (token < logits.cols) {
                if (logits(last_pos, token) > 0) {
                    logits(last_pos, token) /= penalty;
                } else {
                    logits(last_pos, token) *= penalty;
                }
            }
        }
    }

    int sample_token(const Tensor2D& logits, const SamplingConfig& config) const {
        int T = logits.rows;
        int last_pos = T - 1;
        int vocab_size = logits.cols;

        std::vector<float> probs(vocab_size);
        for (int i = 0; i < vocab_size; ++i) {
            probs[i] = logits(last_pos, i);
        }

        // 온도 적용
        if (config.temperature != 1.0f) {
            for (auto& p : probs) {
                p /= config.temperature;
            }
        }

        // Softmax
        float max_logit = *std::max_element(probs.begin(), probs.end());
        float sum_exp = 0.0f;
        for (auto& p : probs) {
            p = std::exp(p - max_logit);
            sum_exp += p;
        }
        for (auto& p : probs) {
            p /= sum_exp;
        }

        switch (config.strategy) {
            case SamplingStrategy::GREEDY:
                return std::max_element(probs.begin(), probs.end()) - probs.begin();
            
            case SamplingStrategy::TOP_K:
                return sample_top_k(probs, config.top_k);
            
            case SamplingStrategy::TOP_P:
                return sample_top_p(probs, config.top_p);
            
            default:
                return sample_multinomial(probs);
        }
    }

    int sample_top_k(std::vector<float>& probs, int k) const {
        // 상위 k개만 남기고 나머지는 0으로
        std::vector<std::pair<float, int>> prob_idx;
        for (int i = 0; i < probs.size(); ++i) {
            prob_idx.emplace_back(probs[i], i);
        }
        
        std::sort(prob_idx.rbegin(), prob_idx.rend());
        
        std::fill(probs.begin(), probs.end(), 0.0f);
        float sum = 0.0f;
        for (int i = 0; i < std::min(k, (int)prob_idx.size()); ++i) {
            probs[prob_idx[i].second] = prob_idx[i].first;
            sum += prob_idx[i].first;
        }
        
        // 재정규화
        for (auto& p : probs) p /= sum;
        
        return sample_multinomial(probs);
    }

    int sample_top_p(std::vector<float>& probs, float p) const {
        std::vector<std::pair<float, int>> prob_idx;
        for (int i = 0; i < probs.size(); ++i) {
            prob_idx.emplace_back(probs[i], i);
        }
        
        std::sort(prob_idx.rbegin(), prob_idx.rend());
        
        std::fill(probs.begin(), probs.end(), 0.0f);
        float cumsum = 0.0f;
        for (const auto& [prob, idx] : prob_idx) {
            probs[idx] = prob;
            cumsum += prob;
            if (cumsum >= p) break;
        }
        
        // 재정규화
        for (auto& prob : probs) prob /= cumsum;
        
        return sample_multinomial(probs);
    }

    int sample_multinomial(const std::vector<float>& probs) const {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        
        float r = dis(gen);
        float cumsum = 0.0f;
        
        for (int i = 0; i < probs.size(); ++i) {
            cumsum += probs[i];
            if (r <= cumsum) return i;
        }
        
        return probs.size() - 1; // fallback
    }
};

} // namespace mt
