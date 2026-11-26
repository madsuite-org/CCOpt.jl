using Documenter
using CCOpt
using DocumenterInterLinks

links = InterLinks(
    "MPCCModels" => "https://madsuite.org/MPCCModels.jl/dev/",
    "MadNLP" => "https://madsuite.org/MadNLP.jl/dev/",
    "JuMP" => "https://jump.dev/JuMP.jl/stable/",
)
makedocs(
    sitename = "CCOpt.jl",
    format = Documenter.HTML(
        assets = ["assets/favicon.ico"],
        prettyurls = Base.get(ENV, "CI", nothing) == "true",
        mathengine = Documenter.KaTeX(),
    ),
    modules = [CCOpt],
    repo = "https://github.com/madsuite-org/CCOpt.jl/blob/{commit}{path}#{line}",
    checkdocs = :none,
    clean = true,
    pages = [
        "Home" => "index.md",
        "Installation" => "installation.md",
        "List of Options" => "options.md",
        "CasADi Tutorial" => "casadi_tutorial.md",
        "JuMP Tutorial" => "jump_tutorial.md",
    ],
    plugins = [links],
)

deploydocs(
    repo = "github.com/madsuite-org/CCOpt.jl.git",
    target = "build",
    devbranch = "master",
    devurl = "dev",
    push_preview = true,
)
