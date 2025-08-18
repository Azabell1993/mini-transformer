#pragma once
#include <model/tensor.hpp>
#include <model/layernorm.hpp>
#include <model/attention.hpp>
#include <model/ffn.hpp>
#include <random>

namespace mt {

// 하나의 트랜스포머 블록: Pre-LN -> MHA -> Residual -> Pre-LN -> FFN -> Residual
struct Block {
    LayerNorm ln1, ln2;
    MultiHeadSelfAttention mha;
    FFN ffn;

    Block() = default;
    Block(int d_model, int n_heads, int d_ff): ln1(d_model), ln2(d_model), mha(d_model, n_heads), ffn(d_model, d_ff) {}

    Tensor2D forward(const Tensor2D& X) const {
        Tensor2D h = X; // 입력을 복사하여 잔차 연결(Residual)에 사용
        Tensor2D x1 = h; ln1.forward_inplace(x1);
        Tensor2D attn = mha.forward(x1);
        mt::add_inplace(h, attn); // Residual Add
        Tensor2D x2 = h; ln2.forward_inplace(x2);
        Tensor2D ff = ffn.forward(x2);
        mt::add_inplace(h, ff);   // Residual Add
        return h;
    }
};

// 전체 트랜스포머 네트워크(디코더 전용, GPT류 단순화)
struct Transformer {
    int vocab{}, d_model{}, n_layers{}, n_heads{}, d_ff{}, max_seq{};
    Tensor2D tok_emb;   // (vocab x d_model) 토큰 임베딩 테이블
    Tensor2D pos_emb;   // (max_seq x d_model) 위치 임베딩 테이블(학습형)
    std::vector<Block> blocks; // 반복 블록
    LayerNorm ln_f;     // 최종 LayerNorm
    
    // Wout (output projection matrix)
    Tensor2D Wout;      // (d_model x vocab) 출력 투영(로짓 생성) 


    Transformer() = default;
    Transformer(int vocab_, int d_model_, int n_layers_, int n_heads_, int d_ff_, int max_seq_)
        : vocab(vocab_), d_model(d_model_), n_layers(n_layers_), n_heads(n_heads_), d_ff(d_ff_), max_seq(max_seq_),
          tok_emb(vocab_, d_model_), pos_emb(max_seq_, d_model_), blocks(n_layers_), ln_f(d_model_), Wout(d_model_, vocab_) {
        for (int i=0;i<n_layers_;++i) blocks[i] = Block(d_model_, n_heads_, d_ff_);
        init_params();
    }

    // 파라미터를 정규분포(평균 0, 표준편차 0.02)로 초기화합니다.
    void init_params(unsigned seed=42) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> nd(0.f, 0.02f);
        auto init = [&](Tensor2D& T){ for (auto& v: T.data) v = nd(rng); };
        init(tok_emb); init(pos_emb); init(Wout);
        for (auto& b: blocks) { init(b.mha.Wq); init(b.mha.Wk); init(b.mha.Wv); init(b.mha.Wo); init(b.ffn.W1); init(b.ffn.W2); }
    }

    // 토큰 ID 시퀀스를 임베딩으로 변환합니다.
    Tensor2D embed(const std::vector<int>& tokens) const {
        int T = tokens.size();
        Tensor2D X(T, d_model);
        for (int t=0;t<T;++t) {
            int id = tokens[t]%vocab; // 범위를 벗어날 경우를 대비해 mod 사용(실제 시스템에선 검증 필요)
            for (int j=0;j<d_model;++j) X(t,j) = tok_emb(id,j) + pos_emb(t,j);
        }
        return X;
    }

    // 순전파: 로짓(T x vocab) 반환
    Tensor2D forward(const std::vector<int>& tokens) const {
        Tensor2D X = embed(tokens);
        for (int i=0;i<n_layers;++i) X = blocks[i].forward(X);
        ln_f.forward_inplace(X);
        Tensor2D logits; matmul(X, Wout, logits);
        return logits;
    }

    // 가장 확률이 높은(로짓 최대) 다음 토큰 인덱스를 반환하는 간단한 선택기
    int next_token_argmax(const std::vector<int>& tokens) const {
        auto logits = forward(tokens);
        int T = logits.rows; int last = T-1;
        float best=-1e30f; int idx=0;
        for (int v=0; v<vocab; ++v) { if (logits(last,v)>best) { best=logits(last,v); idx=v; } }
        return idx;
    }
};

} // namespace mt
