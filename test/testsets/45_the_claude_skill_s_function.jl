@testset "45. The Claude skill's function reference matches the docstrings" begin
    _progress("45. The Claude skill's function reference matches the docstrings")
    # skills/spatial-econometrics/references/api/ is generated from the
    # docstrings by skills/build_reference.jl. A docstring changed without
    # rerunning the script fails here.
    include(joinpath(@__DIR__, "..", "..", "skills", "build_reference.jl"))
    @test isempty(stale_reference_files())
    # Power: a changed, a missing and an extra file are each caught.
    files45 = build_reference()
    mktempdir() do d
        for (f, txt) in files45
            write(joinpath(d, f), txt)
        end
        @test isempty(stale_reference_files(d))
        write(joinpath(d, "SpatialGARCH.md"), files45["SpatialGARCH.md"] * "x")
        rm(joinpath(d, "INDEX.md"))
        write(joinpath(d, "Retired.md"), "")
        @test stale_reference_files(d) == ["INDEX.md", "Retired.md", "SpatialGARCH.md"]
    end
end
