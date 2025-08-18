#pragma once
#include <vector>
#include <cmath>
#include <cassert>
#include <algorithm>

namespace mt {

// 매우 단순한 2차원 텐서(행렬) 클래스입니다.
// - 행 우선(Row-major) 1차원 버퍼에 데이터를 저장합니다.
// - 실습 목적이므로 범용 텐서 라이브러리 대신 최소 기능만 제공합니다.
struct Tensor2D {
    int rows{0}, cols{0};
    std::vector<float> data;

    Tensor2D() = default;
    Tensor2D(int r, int c) : rows(r), cols(c), data(r*c) {}

    float* row(int r) { return data.data() + r*cols; }
    const float* row(int r) const { return data.data() + r*cols; }

    float& operator()(int r, int c){ return data[r*cols + c]; }
    float operator()(int r, int c) const { return data[r*cols + c]; }
};

// 행렬 곱셈 C = A * B (순진한 3중 for문)
// - A: (m x k), B: (k x n), C: (m x n)
inline void matmul(const Tensor2D& A, const Tensor2D& B, Tensor2D& C) {
    assert(A.cols == B.rows);
    C = Tensor2D(A.rows, B.cols);
    for (int i=0;i<A.rows;++i) {
        for (int k=0;k<A.cols;++k) {
            float a = A(i,k);
            for (int j=0;j<B.cols;++j) {
                C(i,j) += a * B(k,j);
            }
        }
    }
}

// A += B (원소별 덧셈)
inline void add_inplace(Tensor2D& A, const Tensor2D& B) {
    assert(A.rows==B.rows && A.cols==B.cols);
    for (size_t i=0;i<A.data.size();++i) A.data[i]+=B.data[i];
}

// 각 행(row)에 대해 소프트맥스를 적용합니다.
// - 수치 안정성을 위해 행 최대값을 빼준 후 exp를 계산합니다.
inline void softmax_rows(Tensor2D& X) {
    for (int i=0;i<X.rows;++i) {
        float m = -1e30f;
        for (int j=0;j<X.cols;++j) m = std::max(m, X(i,j));
        float sum=0;
        for (int j=0;j<X.cols;++j) { X(i,j)=std::exp(X(i,j)-m); sum+=X(i,j);} 
        for (int j=0;j<X.cols;++j) X(i,j)/=sum; 
    }
}

} // namespace mt
