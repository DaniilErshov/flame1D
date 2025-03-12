# CMake generated Testfile for 
# Source directory: D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial
# Build directory: D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials_build/examples/kinsol/serial
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
if(CTEST_CONFIGURATION_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test(kinAnalytic_fp "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials_build/bin/Debug/kinAnalytic_fp.exe")
  set_tests_properties(kinAnalytic_fp PROPERTIES  _BACKTRACE_TRIPLES "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/cmake/macros/SundialsAddTest.cmake;183;add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;98;sundials_add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test(kinAnalytic_fp "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials_build/bin/Release/kinAnalytic_fp.exe")
  set_tests_properties(kinAnalytic_fp PROPERTIES  _BACKTRACE_TRIPLES "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/cmake/macros/SundialsAddTest.cmake;183;add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;98;sundials_add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test(kinAnalytic_fp "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials_build/bin/MinSizeRel/kinAnalytic_fp.exe")
  set_tests_properties(kinAnalytic_fp PROPERTIES  _BACKTRACE_TRIPLES "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/cmake/macros/SundialsAddTest.cmake;183;add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;98;sundials_add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test(kinAnalytic_fp "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials_build/bin/RelWithDebInfo/kinAnalytic_fp.exe")
  set_tests_properties(kinAnalytic_fp PROPERTIES  _BACKTRACE_TRIPLES "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/cmake/macros/SundialsAddTest.cmake;183;add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;98;sundials_add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;0;")
else()
  add_test(kinAnalytic_fp NOT_AVAILABLE)
endif()
if(CTEST_CONFIGURATION_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test(kinAnalytic_fp_--m_aa_2 "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials_build/bin/Debug/kinAnalytic_fp.exe" "--m_aa" "2")
  set_tests_properties(kinAnalytic_fp_--m_aa_2 PROPERTIES  _BACKTRACE_TRIPLES "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/cmake/macros/SundialsAddTest.cmake;183;add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;98;sundials_add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test(kinAnalytic_fp_--m_aa_2 "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials_build/bin/Release/kinAnalytic_fp.exe" "--m_aa" "2")
  set_tests_properties(kinAnalytic_fp_--m_aa_2 PROPERTIES  _BACKTRACE_TRIPLES "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/cmake/macros/SundialsAddTest.cmake;183;add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;98;sundials_add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test(kinAnalytic_fp_--m_aa_2 "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials_build/bin/MinSizeRel/kinAnalytic_fp.exe" "--m_aa" "2")
  set_tests_properties(kinAnalytic_fp_--m_aa_2 PROPERTIES  _BACKTRACE_TRIPLES "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/cmake/macros/SundialsAddTest.cmake;183;add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;98;sundials_add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test(kinAnalytic_fp_--m_aa_2 "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials_build/bin/RelWithDebInfo/kinAnalytic_fp.exe" "--m_aa" "2")
  set_tests_properties(kinAnalytic_fp_--m_aa_2 PROPERTIES  _BACKTRACE_TRIPLES "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/cmake/macros/SundialsAddTest.cmake;183;add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;98;sundials_add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;0;")
else()
  add_test(kinAnalytic_fp_--m_aa_2 NOT_AVAILABLE)
endif()
if(CTEST_CONFIGURATION_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test(kinAnalytic_fp_--m_aa_2_--delay_aa_2 "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials_build/bin/Debug/kinAnalytic_fp.exe" "--m_aa" "2" "--delay_aa" "2")
  set_tests_properties(kinAnalytic_fp_--m_aa_2_--delay_aa_2 PROPERTIES  _BACKTRACE_TRIPLES "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/cmake/macros/SundialsAddTest.cmake;183;add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;98;sundials_add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test(kinAnalytic_fp_--m_aa_2_--delay_aa_2 "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials_build/bin/Release/kinAnalytic_fp.exe" "--m_aa" "2" "--delay_aa" "2")
  set_tests_properties(kinAnalytic_fp_--m_aa_2_--delay_aa_2 PROPERTIES  _BACKTRACE_TRIPLES "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/cmake/macros/SundialsAddTest.cmake;183;add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;98;sundials_add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test(kinAnalytic_fp_--m_aa_2_--delay_aa_2 "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials_build/bin/MinSizeRel/kinAnalytic_fp.exe" "--m_aa" "2" "--delay_aa" "2")
  set_tests_properties(kinAnalytic_fp_--m_aa_2_--delay_aa_2 PROPERTIES  _BACKTRACE_TRIPLES "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/cmake/macros/SundialsAddTest.cmake;183;add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;98;sundials_add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test(kinAnalytic_fp_--m_aa_2_--delay_aa_2 "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials_build/bin/RelWithDebInfo/kinAnalytic_fp.exe" "--m_aa" "2" "--delay_aa" "2")
  set_tests_properties(kinAnalytic_fp_--m_aa_2_--delay_aa_2 PROPERTIES  _BACKTRACE_TRIPLES "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/cmake/macros/SundialsAddTest.cmake;183;add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;98;sundials_add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;0;")
else()
  add_test(kinAnalytic_fp_--m_aa_2_--delay_aa_2 NOT_AVAILABLE)
endif()
if(CTEST_CONFIGURATION_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test(kinAnalytic_fp_--m_aa_2_--orth_aa_1 "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials_build/bin/Debug/kinAnalytic_fp.exe" "--m_aa" "2" "--orth_aa" "1")
  set_tests_properties(kinAnalytic_fp_--m_aa_2_--orth_aa_1 PROPERTIES  _BACKTRACE_TRIPLES "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/cmake/macros/SundialsAddTest.cmake;183;add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;98;sundials_add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test(kinAnalytic_fp_--m_aa_2_--orth_aa_1 "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials_build/bin/Release/kinAnalytic_fp.exe" "--m_aa" "2" "--orth_aa" "1")
  set_tests_properties(kinAnalytic_fp_--m_aa_2_--orth_aa_1 PROPERTIES  _BACKTRACE_TRIPLES "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/cmake/macros/SundialsAddTest.cmake;183;add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;98;sundials_add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test(kinAnalytic_fp_--m_aa_2_--orth_aa_1 "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials_build/bin/MinSizeRel/kinAnalytic_fp.exe" "--m_aa" "2" "--orth_aa" "1")
  set_tests_properties(kinAnalytic_fp_--m_aa_2_--orth_aa_1 PROPERTIES  _BACKTRACE_TRIPLES "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/cmake/macros/SundialsAddTest.cmake;183;add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;98;sundials_add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test(kinAnalytic_fp_--m_aa_2_--orth_aa_1 "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials_build/bin/RelWithDebInfo/kinAnalytic_fp.exe" "--m_aa" "2" "--orth_aa" "1")
  set_tests_properties(kinAnalytic_fp_--m_aa_2_--orth_aa_1 PROPERTIES  _BACKTRACE_TRIPLES "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/cmake/macros/SundialsAddTest.cmake;183;add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;98;sundials_add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;0;")
else()
  add_test(kinAnalytic_fp_--m_aa_2_--orth_aa_1 NOT_AVAILABLE)
endif()
if(CTEST_CONFIGURATION_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test(kinAnalytic_fp_--m_aa_2_--orth_aa_2 "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials_build/bin/Debug/kinAnalytic_fp.exe" "--m_aa" "2" "--orth_aa" "2")
  set_tests_properties(kinAnalytic_fp_--m_aa_2_--orth_aa_2 PROPERTIES  _BACKTRACE_TRIPLES "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/cmake/macros/SundialsAddTest.cmake;183;add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;98;sundials_add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test(kinAnalytic_fp_--m_aa_2_--orth_aa_2 "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials_build/bin/Release/kinAnalytic_fp.exe" "--m_aa" "2" "--orth_aa" "2")
  set_tests_properties(kinAnalytic_fp_--m_aa_2_--orth_aa_2 PROPERTIES  _BACKTRACE_TRIPLES "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/cmake/macros/SundialsAddTest.cmake;183;add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;98;sundials_add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test(kinAnalytic_fp_--m_aa_2_--orth_aa_2 "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials_build/bin/MinSizeRel/kinAnalytic_fp.exe" "--m_aa" "2" "--orth_aa" "2")
  set_tests_properties(kinAnalytic_fp_--m_aa_2_--orth_aa_2 PROPERTIES  _BACKTRACE_TRIPLES "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/cmake/macros/SundialsAddTest.cmake;183;add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;98;sundials_add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test(kinAnalytic_fp_--m_aa_2_--orth_aa_2 "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials_build/bin/RelWithDebInfo/kinAnalytic_fp.exe" "--m_aa" "2" "--orth_aa" "2")
  set_tests_properties(kinAnalytic_fp_--m_aa_2_--orth_aa_2 PROPERTIES  _BACKTRACE_TRIPLES "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/cmake/macros/SundialsAddTest.cmake;183;add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;98;sundials_add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;0;")
else()
  add_test(kinAnalytic_fp_--m_aa_2_--orth_aa_2 NOT_AVAILABLE)
endif()
if(CTEST_CONFIGURATION_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test(kinAnalytic_fp_--m_aa_2_--orth_aa_3 "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials_build/bin/Debug/kinAnalytic_fp.exe" "--m_aa" "2" "--orth_aa" "3")
  set_tests_properties(kinAnalytic_fp_--m_aa_2_--orth_aa_3 PROPERTIES  _BACKTRACE_TRIPLES "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/cmake/macros/SundialsAddTest.cmake;183;add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;98;sundials_add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test(kinAnalytic_fp_--m_aa_2_--orth_aa_3 "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials_build/bin/Release/kinAnalytic_fp.exe" "--m_aa" "2" "--orth_aa" "3")
  set_tests_properties(kinAnalytic_fp_--m_aa_2_--orth_aa_3 PROPERTIES  _BACKTRACE_TRIPLES "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/cmake/macros/SundialsAddTest.cmake;183;add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;98;sundials_add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test(kinAnalytic_fp_--m_aa_2_--orth_aa_3 "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials_build/bin/MinSizeRel/kinAnalytic_fp.exe" "--m_aa" "2" "--orth_aa" "3")
  set_tests_properties(kinAnalytic_fp_--m_aa_2_--orth_aa_3 PROPERTIES  _BACKTRACE_TRIPLES "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/cmake/macros/SundialsAddTest.cmake;183;add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;98;sundials_add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test(kinAnalytic_fp_--m_aa_2_--orth_aa_3 "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials_build/bin/RelWithDebInfo/kinAnalytic_fp.exe" "--m_aa" "2" "--orth_aa" "3")
  set_tests_properties(kinAnalytic_fp_--m_aa_2_--orth_aa_3 PROPERTIES  _BACKTRACE_TRIPLES "D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/cmake/macros/SundialsAddTest.cmake;183;add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;98;sundials_add_test;D:/Storage/Daniil/gorenie_version/gorenie_try1_10/sundials/sundials-6.5.1/examples/kinsol/serial/CMakeLists.txt;0;")
else()
  add_test(kinAnalytic_fp_--m_aa_2_--orth_aa_3 NOT_AVAILABLE)
endif()
