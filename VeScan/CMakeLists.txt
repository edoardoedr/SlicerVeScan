#-----------------------------------------------------------------------------
set(MODULE_NAME VeScan)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/UI/${MODULE_NAME}.ui
  Resources/vescan_field.json
  Resources/vescan_thalweg.json
  Resources/ColorTables/ColorTableCrossings_EN.csv
  Resources/ColorTables/ColorTableLobeLabel_EN.csv
  Resources/ColorTables/ColorTableLobesFissures_EN.csv
  Resources/ColorTables/ColorTableVessels_EN.csv
  Resources/requirements.txt
  Resources/requirements-deps.txt
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)

  # Register the unittest subclass in the main script as a ctest.
  # Note that the test will also be available at runtime.
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)

  # Additional build-time testing
  add_subdirectory(Testing)
endif()
