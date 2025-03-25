@echo off
set oldPath=%cd%


:: 设置 MSVC 编译器环境变量（路径根据你的安装位置调整）
call "D:\VisualStudio\visualstudio\VC\Auxiliary\Build\vcvars64.bat"
:: 设置项目路径
set PROJECT_ROOT=%~dp0
cd /d "%PROJECT_ROOT%"

:: 清理旧的构建目录
if exist build rmdir /s /q build


@REM cd /d %~dp0
cmake -S . -B build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build

cd  /d %oldPath%