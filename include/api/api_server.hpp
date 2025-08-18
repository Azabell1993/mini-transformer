#pragma once
#include <string>
#include <memory>
#include <thread>
#include <atomic>
#include <functional>
#include <vector>

namespace api {

// 간단한 HTTP 서버(Boost.Beast 기반) 스텁 클래스입니다.
// - init(): 초기화
// - start(): 별도 스레드에서 동기 accept 루프를 돌며 요청을 처리
// - stop(): 루프 종료 및 스레드 합류
// - setPredictHandler(): "/predict" 요청 처리 콜백 등록(토큰 벡터 -> HTML 스니펫)
class ApiServer {
public:
    using PredictHandler = std::function<std::string(const std::vector<int>&)>;

    ApiServer(const std::string& host, int port, const std::string& docRoot = "");
    ~ApiServer();

    void init();  // 서버 준비 작업(필요 시 리소스 준비)
    void start(); // 서버 시작(accept 루프)
    void stop();  // 서버 정지

    void setPredictHandler(PredictHandler handler) { m_predict = std::move(handler); }

private:
    std::string m_host;       // 바인딩할 호스트 주소(예: "0.0.0.0")
    int m_port;               // 포트 번호(예: 18080)
    std::string m_doc_root;   // 정적 파일 루트(예: web/)
    std::atomic<bool> m_running{false}; // 실행 플래그
    std::thread m_thread;     // 서버 실행 스레드
    PredictHandler m_predict; // 예측 핸들러(옵션)
};

} // namespace api
