#pragma once
#include <model/tensor.hpp>
#include <vector>

namespace mt {

// 멀티-헤드 셀프 어텐션(Multi-Head Self-Attention)
// - 입력 X(T x d_model)에서 Q,K,V를 생성하고, 각 헤드별로 어텐션을 계산합니다.
// - 점수 = QK^T / sqrt(d_head), 행별 softmax, 가중합 * V 순서로 진행합니다.
// - 모든 헤드 출력을 concat하여 W_o로 투영합니다.
struct MultiHeadSelfAttention {
    int d_model{}, n_heads{}, d_head{};
    bool use_causal_mask = false;  // [TODO] 마스크 사용 플래그 (기본 false)
    Tensor2D Wq, Wk, Wv, Wo; // (d_model x d_model)

    MultiHeadSelfAttention() = default;
    MultiHeadSelfAttention(int d_model_, int n_heads_): d_model(d_model_), n_heads(n_heads_) {
        d_head = d_model / n_heads;
        Wq = Tensor2D(d_model, d_model);
        Wk = Tensor2D(d_model, d_model);
        Wv = Tensor2D(d_model, d_model);
        Wo = Tensor2D(d_model, d_model);
    }

    // X: (T x d_model), Pre-LN 입력을 가정
    Tensor2D forward(const Tensor2D& X) const {
        Tensor2D Q, K, V;
        matmul(X, Wq, Q); // (T x d_model)
        matmul(X, Wk, K);
        matmul(X, Wv, V);

        int T = X.rows;
        Tensor2D out(T, d_model);
        // 각 헤드에 대해 부분 행렬을 추출하여 어텐션 계산
        for (int h=0; h<n_heads; ++h) {
            auto slice = [&](const Tensor2D& A)->Tensor2D{
                Tensor2D S(T, d_head);
                for (int t=0;t<T;++t)
                    for (int i=0;i<d_head;++i)
                        S(t,i) = A(t, h*d_head + i);
                return S;
            };
            Tensor2D Qh = slice(Q), Kh = slice(K), Vh = slice(V);

            // 점수 행렬 = Qh * Kh^T / sqrt(d_head)
            Tensor2D KhT(Kh.cols, Kh.rows);
            for (int i=0;i<Kh.rows;++i)
                for (int j=0;j<Kh.cols;++j)
                    KhT(j,i) = Kh(i,j);
            Tensor2D scores; matmul(Qh, KhT, scores);
            float scale = 1.0f/std::sqrt((float)d_head);
            for (auto& v: scores.data) v *= scale;

            /**
             * Causal Masking
             * 
             * - 목적: 미래 토큰(j > t)을 보지 못하도록 상삼각 영역에 큰 음수(-1e9f) 더하기
             * - 형태: for t in [0..T): for j in [t+1..T): scores(t,j) += NEG_INF;
             * - 주의: softmax_rows(scores) "바로 이전"에 수행되어야 함
             * - 성능: O(T^2). T가 작을 때는 OK, 크면 블록/벡터화 고려
             */
            if (use_causal_mask) {
                const float NEG_INF = -1e9f; // 또는 -std::numeric_limits<float>::infinity()
                for (int t=0; t<T; ++t) {
                    for (int j=t+1; j<T; ++j) {
                        scores(t, j) += NEG_INF;
                    }
                }
            }

            // 행렬 softmax (tensor.hpp에서 수치 안전성 지원 - row-max 감산 후 exp)
            softmax_rows(scores);

            // 어텐션 출력 = softmax(scores) * Vh
            Tensor2D head_out; matmul(scores, Vh, head_out); // (T x d_head)
            // concat 결과를 out에 배치
            for (int t=0;t<T;++t)
                for (int i=0;i<d_head;++i)
                    out(t, h*d_head + i) = head_out(t,i);
        }
        // 최종 투영
        Tensor2D proj; matmul(out, Wo, proj); // (T x d_model)
        return proj;
    }
};

} // namespace mt
