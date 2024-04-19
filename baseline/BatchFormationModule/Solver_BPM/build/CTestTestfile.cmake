# CMake generated Testfile for 
# Source directory: D:/ORTools/or-tools_x64_VisualStudio2022_cpp_v9.9.3963/examples/Solver_BPM
# Build directory: D:/ORTools/or-tools_x64_VisualStudio2022_cpp_v9.9.3963/examples/Solver_BPM/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
if(CTEST_CONFIGURATION_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test(test_Solver_BPM "D:/ORTools/or-tools_x64_VisualStudio2022_cpp_v9.9.3963/examples/Solver_BPM/build/Debug/bin/Solver_BPM.exe")
  set_tests_properties(test_Solver_BPM PROPERTIES  _BACKTRACE_TRIPLES "D:/ORTools/or-tools_x64_VisualStudio2022_cpp_v9.9.3963/examples/Solver_BPM/CMakeLists.txt;81;add_test;D:/ORTools/or-tools_x64_VisualStudio2022_cpp_v9.9.3963/examples/Solver_BPM/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test(test_Solver_BPM "D:/ORTools/or-tools_x64_VisualStudio2022_cpp_v9.9.3963/examples/Solver_BPM/build/Release/bin/Solver_BPM.exe")
  set_tests_properties(test_Solver_BPM PROPERTIES  _BACKTRACE_TRIPLES "D:/ORTools/or-tools_x64_VisualStudio2022_cpp_v9.9.3963/examples/Solver_BPM/CMakeLists.txt;81;add_test;D:/ORTools/or-tools_x64_VisualStudio2022_cpp_v9.9.3963/examples/Solver_BPM/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test(test_Solver_BPM "D:/ORTools/or-tools_x64_VisualStudio2022_cpp_v9.9.3963/examples/Solver_BPM/build/MinSizeRel/bin/Solver_BPM.exe")
  set_tests_properties(test_Solver_BPM PROPERTIES  _BACKTRACE_TRIPLES "D:/ORTools/or-tools_x64_VisualStudio2022_cpp_v9.9.3963/examples/Solver_BPM/CMakeLists.txt;81;add_test;D:/ORTools/or-tools_x64_VisualStudio2022_cpp_v9.9.3963/examples/Solver_BPM/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test(test_Solver_BPM "D:/ORTools/or-tools_x64_VisualStudio2022_cpp_v9.9.3963/examples/Solver_BPM/build/RelWithDebInfo/bin/Solver_BPM.exe")
  set_tests_properties(test_Solver_BPM PROPERTIES  _BACKTRACE_TRIPLES "D:/ORTools/or-tools_x64_VisualStudio2022_cpp_v9.9.3963/examples/Solver_BPM/CMakeLists.txt;81;add_test;D:/ORTools/or-tools_x64_VisualStudio2022_cpp_v9.9.3963/examples/Solver_BPM/CMakeLists.txt;0;")
else()
  add_test(test_Solver_BPM NOT_AVAILABLE)
endif()
