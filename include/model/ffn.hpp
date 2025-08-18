#pragma once
#include <model/tensor.hpp>

namespace mt {

// 위치별 FFN(2층 MLP)
// - 각 토큰 위치(행) 독립적으로 W1 확장 -> 활성함수(GELU) -> W2 축소를 적용합니다.
struct FFN {
    int d_model{}, d_ff{};
    Tensor2D W1; // (d_model x d_ff)
    Tensor2D W2; // (d_ff x d_model)

    FFN() = default;
    FFN(int dm, int df): d_model(dm), d_ff(df), W1(dm, df), W2(df, dm) {}

    // GELU 근사식(tanh 버전)
    static inline float gelu(float x) {
        const float kBeta = 0.044715f;
        return 0.5f * x * (1.0f + std::tanh(std::sqrt(2.0f/M_PI)*(x + kBeta*x*x*x)));
    }

    Tensor2D forward(const Tensor2D& X) const {
        Tensor2D H; matmul(X, W1, H); // (T x d_ff)
        for (int i=0;i<H.rows;++i)
            for (int j=0;j<H.cols;++j)
                H(i,j) = gelu(H(i,j));
        Tensor2D O; matmul(H, W2, O); // (T x d_model)
        return O;
    }
};

} // namespace mt
