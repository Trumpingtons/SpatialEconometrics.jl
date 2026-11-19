@testset "18b. Table exporters escape and honour digits (finding 23)" begin
    _progress("18b. Table exporters escape and honour digits (finding 23)")
    ST = SpatialEconometrics.SpatialTables
    W = make_rook_weights(8); N = size(W, 1)
    rg = MersenneTwister(31); X = randn(rg, N, 2)
    lab = [i <= N ÷ 2 ? 1 : 2 for i in 1:N]
    y = (I - 0.3 * Matrix(W)) \ (1.0 .+ X * [1.0, -0.5] .+ 0.6 .* randn(rg, N))
    # Regime names carry underscores, and ρ is Greek — between them they
    # exercise every escape path the exporters need.
    m = fit_spatial_regime(W, X, y, ExogenousRegimes(lab); varnames=["x1", "x2"])
    ct = coeftable(m)
    @test any(nm -> occursin("_", nm), ct.names)
    @test "ρ" in ct.names

    # Count the pipes a Markdown renderer treats as cell separators, i.e.
    # those not escaped by a preceding backslash.
    function sep_pipes(line::AbstractString)
        n = 0
        prev = ' '
        for c in line
            c == '|' && prev != '\\' && (n += 1)
            prev = c
        end
        return n
    end

    # ── to_markdown: every row must declare the same cells ──
    md = to_markdown(ct)
    rows = filter(!isempty, split(strip(md), '\n'))
    @test length(rows) == length(ct.names) + 2        # header + rule + body
    want = sep_pipes(rows[2])                          # the separator row
    @test want == 7                                    # 6 columns
    @test all(r -> sep_pipes(r) == want, rows)
    # The header's P>|z| must be escaped, not bare.
    @test occursin("P>\\|z\\|", rows[1])
    # A name containing a pipe must not break its row either.
    ct_pipe = ST.CoefTable{Float64}(["a|b"], [1.0], [0.1], [10.0], [0.0],
                                    [0.8], [1.2], 0.95)
    rows_p = filter(!isempty, split(strip(to_markdown(ct_pipe)), '\n'))
    @test all(r -> sep_pipes(r) == want, rows_p)

    # ── to_latex: no bare specials, Greek in math mode ──
    tex = to_latex(ct)
    body = filter(l -> occursin("&", l) && !occursin("Variable", l),
                  split(tex, '\n'))
    @test length(body) == length(ct.names)
    for l in body
        nm = strip(first(split(l, '&')))
        # Every underscore in a name must be backslash-escaped.
        for (i, c) in enumerate(nm)
            c == '_' && @test i > 1 && nm[prevind(nm, i)] == '\\'
        end
    end
    @test occursin("\$\\rho\$", tex)       # ρ became $\rho$
    @test !occursin("ρ", tex)              # and no raw Greek is left
    @test occursin("\\begin{tabular}", tex) && occursin("\\end{tabular}", tex)
    @test ST._escape_latex("a_b&c%d#e") == "a\\_b\\&c\\%d\\#e"
    @test ST._escape_latex("plain") == "plain"

    # ── to_csv: RFC 4180 quoting ──
    ct_c = ST.CoefTable{Float64}(["a,b", "q\"r", "plain"], [1.0, 2.0, 3.0],
        [0.1, 0.2, 0.3], [10.0, 10.0, 10.0], [0.0, 0.0, 0.0],
        [0.8, 1.6, 2.4], [1.2, 2.4, 3.6], 0.95)
    csv = to_csv(ct_c)
    # A minimal RFC 4180 field splitter: quotes toggle, commas inside them
    # are data. Each row must come out at exactly the 7-column schema.
    function rfc_fields(line::AbstractString)
        fields = String[]; buf = IOBuffer(); inq = false; i = 1
        chars = collect(line)
        while i <= length(chars)
            c = chars[i]
            if inq
                if c == '"'
                    if i < length(chars) && chars[i + 1] == '"'
                        print(buf, '"'); i += 1
                    else
                        inq = false
                    end
                else
                    print(buf, c)
                end
            elseif c == '"'
                inq = true
            elseif c == ','
                push!(fields, String(take!(buf)))
            else
                print(buf, c)
            end
            i += 1
        end
        push!(fields, String(take!(buf)))
        return fields
    end
    lines = filter(!isempty, split(strip(csv), '\n'))
    @test length(lines) == 4                       # header + 3 rows
    @test all(l -> length(rfc_fields(l)) == 7, lines)
    # …and the names survive the round trip intact.
    @test [rfc_fields(l)[1] for l in lines[2:end]] == ["a,b", "q\"r", "plain"]
    @test ST._csv_field("plain") == "plain"        # untouched when safe

    # ── `digits` actually reaches the output ──
    # Count the decimals in the estimate column at two settings.
    dec(cell) = (parts = split(strip(cell), '.'); length(parts) == 2 ?
                 length(filter(isdigit, parts[2])) : 0)
    md2 = to_markdown(ct; digits=2); md6 = to_markdown(ct; digits=6)
    @test md2 != md6
    cell2 = split(filter(!isempty, split(strip(md2), '\n'))[3], '|')[3]
    cell6 = split(filter(!isempty, split(strip(md6), '\n'))[3], '|')[3]
    @test dec(cell2) == 2
    @test dec(cell6) == 6
    tex2 = to_latex(ct; digits=2); tex6 = to_latex(ct; digits=6)
    @test tex2 != tex6
    @test dec(split(first(filter(l -> occursin("&", l) &&
             !occursin("Variable", l), split(tex2, '\n'))), '&')[2]) == 2
    @test dec(split(first(filter(l -> occursin("&", l) &&
             !occursin("Variable", l), split(tex6, '\n'))), '&')[2]) == 6
    @test ST._fmt(1.23456, 0) == "1"
    @test ST._fmt(1.23456, 3) == "1.235"

    # ── regression_table matches coefficients by name, not position ──
    # SAR carries ρ and SEM carries λ, so neither name appears in both
    # fits. Each must show exactly one number on its row; borrowing the
    # neighbouring model's figure would put two there.
    m_sar = fit_spatial(SpatialEconometrics.CrossSection.SAR(), W, X, y)
    m_sem = fit_spatial_error(W, X, y)
    rt = regression_table(m_sar, m_sem; model_names=["SAR", "SEM"])
    @test occursin("SAR", rt) && occursin("SEM", rt)
    rt_lines = split(rt, '\n')
    nnum(l) = length(collect(eachmatch(r"-?\d+\.\d+", l)))
    rho_line = first(filter(l -> startswith(l, "ρ"), rt_lines))
    lam_line = first(filter(l -> startswith(l, "λ"), rt_lines))
    @test nnum(rho_line) == 1      # SAR only
    @test nnum(lam_line) == 1      # SEM only
    # A name both models share must show both numbers.
    int_line = first(filter(l -> startswith(l, "(Intercept)"), rt_lines))
    @test nnum(int_line) == 2
end
