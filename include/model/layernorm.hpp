#pragma once
#include <model/tensor.hpp>
#include <vector>

namespace mt {

// Layer Normalization (채널 차원에 대해 평균/분산 정규화)
// - 입력 X의 각 행(토큰 위치)에 대해 채널(d_model) 축 기준 평균/분산을 구합니다.
// - (X - mean) / sqrt(var + eps)로 정규화 후, 학습 파라미터 gamma/beta로 스케일/시프트합니다.
struct LayerNorm {
    int dim{}; float eps{1e-5f};
    std::vector<float> gamma, beta;
    LayerNorm() = default;
    LayerNorm(int d, float e=1e-5f): dim(d), eps(e), gamma(d,1.0f), beta(d,0.0f) {}

    void forward_inplace(Tensor2D& X) const {
        for (int i=0;i<X.rows;++i) {
            float mean=0, var=0; 
            for (int j=0;j<dim;++j) mean += X(i,j);
            mean/=dim;
            for (int j=0;j<dim;++j) { float v=X(i,j)-mean; var += v*v; }
            var/=dim;
            float inv = 1.0f/std::sqrt(var+eps);
            for (int j=0;j<dim;++j) {
                float xn = (X(i,j)-mean)*inv;
                X(i,j) = xn*gamma[j] + beta[j];
            }
        }
    }
};

} // namespace mt
