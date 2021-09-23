include(FetchContent)

FetchContent_Declare(
  knncolle 
  GIT_REPOSITORY https://github.com/LTLA/knncolle
  GIT_TAG ea57cbd4a0ea5574caefd6cd358f232a3813e639
)

FetchContent_MakeAvailable(knncolle)

FetchContent_Declare(
  aarand
  GIT_REPOSITORY https://github.com/LTLA/aarand
  GIT_TAG afb49e269e02000373c55ccc982a4817be2b9d9d
)

FetchContent_MakeAvailable(aarand)
