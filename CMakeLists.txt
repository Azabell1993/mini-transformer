cmake_minimum_required(VERSION 3.16)
project(mini_transformer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# macOS 특정 설정
if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15")
    # Homebrew 경로 추가
    if(EXISTS "/opt/homebrew")
        # Apple Silicon Mac
        set(CMAKE_PREFIX_PATH "/opt/homebrew" ${CMAKE_PREFIX_PATH})
    elseif(EXISTS "/usr/local")
        # Intel Mac
        set(CMAKE_PREFIX_PATH "/usr/local" ${CMAKE_PREFIX_PATH})
    endif()
endif()

option(ENABLE_LICENSE_CHECK "Enable license verification" OFF)

# Boost (for Beast/Asio) - macOS/Linux 호환
find_package(Boost REQUIRED COMPONENTS system)

# JSON 라이브러리 찾기
find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp
    PATHS
        /opt/homebrew/include  # Apple Silicon
        /usr/local/include     # Intel Mac, Linux
        /usr/include           # Linux
    NO_DEFAULT_PATH
)

if(NOT NLOHMANN_JSON_INCLUDE_DIR)
    message(STATUS "nlohmann/json not found in system paths, using find_package")
    find_package(nlohmann_json QUIET)
    if(NOT nlohmann_json_FOUND)
        message(WARNING "nlohmann/json not found. Install with: brew install nlohmann-json (macOS) or apt install nlohmann-json3-dev (Linux)")
    endif()
endif()

add_executable(mini_transformer
    app/main.cpp
    src/engine/engine.cpp
    src/api/api_server.cpp
    src/utils/utils.cpp
    src/secure/secure.cpp
    src/model/transformer.cpp
    src/model/layernorm.cpp
    src/model/attention.cpp
    src/model/ffn.cpp
)

target_include_directories(mini_transformer PRIVATE 
    ${Boost_INCLUDE_DIRS} 
    include
)

# JSON 헤더 경로 추가
if(NLOHMANN_JSON_INCLUDE_DIR)
    target_include_directories(mini_transformer PRIVATE ${NLOHMANN_JSON_INCLUDE_DIR})
endif()

if(ENABLE_LICENSE_CHECK)
    target_compile_definitions(mini_transformer PRIVATE ENABLE_LICENSE_CHECK)
endif()

# macOS에서 컴파일러 최적화
if(APPLE)
    target_compile_options(mini_transformer PRIVATE -Wall -Wextra)
endif()

# nlohmann::json이 pkg-config로 찾아진 경우
if(nlohmann_json_FOUND)
    target_link_libraries(mini_transformer PRIVATE nlohmann_json::nlohmann_json)
endif()

# Link Boost::system and pthread for Asio
find_package(Threads REQUIRED)

# 라이브러리 링크 - 개선된 방식
if(TARGET Boost::system)
    target_link_libraries(mini_transformer PRIVATE Boost::system Threads::Threads)
else()
    target_link_libraries(mini_transformer PRIVATE ${Boost_SYSTEM_LIBRARY} Threads::Threads)
endif()

# Install layout
install(TARGETS mini_transformer RUNTIME DESTINATION bin)
install(DIRECTORY web/ DESTINATION share/mini-transformer/web)
install(DIRECTORY config/ DESTINATION share/mini-transformer/config)
