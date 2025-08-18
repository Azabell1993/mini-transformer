#include <secure/secure.hpp>
#include <fstream>

namespace secure {

// 라이선스 파일을 통째로 읽어 문자열로 반환합니다.
// - 실제 환경에서는 암호화/서명 검증 등을 추가해야 합니다.
bool loadLicenseFile(const std::string& path, std::string& out) {
    std::ifstream ifs(path);
    if (!ifs) return false;
    out.assign((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    return true;
}

// 아래 함수들은 스텁으로, 항상 true를 반환하도록 되어 있습니다.
bool AntiPiracy::verifyProgramIntegrity() { return true; }
bool AntiPiracy::activateOnlineFromJson(const std::string&) { return true; }

bool SignatureVerifier::verifySignatureFromJson(const std::string&, const std::string&) { return true; }

} // namespace secure
