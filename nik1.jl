using CSV, DataFrames, Statistics, GLMakie, Clustering, LinearAlgebra, MultivariateStats, StatsBase, Printf

function print_header(text)
    println("\n" * repeat("=", 60))
    println("=== $text ===")
    println(repeat("=", 60))
end

function main()
    print_header("1. WCZYTYWANIE DANYCH")
    filename = "exoplanet.eu_catalog_20-11-25_14_48_51.csv"

    if !isfile(filename)
        error("Brak pliku '$filename'. Upewnij się, że jest w folderze roboczym.")
    end

    df = CSV.read(filename, DataFrame)

    # Wybieramy zmienne do PCA (Mieszanka cech planety i gwiazdy)
    # Uwaga: Metaliczność to już logarytm ([Fe/H]), więc jej nie logarytmujemy później
    pca_cols = [:mass, :radius, :orbital_period, :star_metallicity, :star_mass, :star_teff]

    # Filtrowanie: Tylko potwierdzone i takie, które mają Masę, Promień i Okres (baza)
    df_clean = df[coalesce.(df.planet_status, "").=="Confirmed", :]

    # Dodajemy kolumnę gęstości przed czyszczeniem (potrzebna do kolorowania)
    # Zamieniamy missing na NaN dla obliczeń
    m = passmissing(Float64).(df_clean.mass)
    r = passmissing(Float64).(df_clean.radius)
    df_clean.density = m ./ (r .^ 3)

    # Usuwamy wiersze, gdzie kluczowe parametry planetarne są missing/NaN
    # (PCA nie zadziała z dziurami)
    dropmissing!(df_clean, [:mass, :radius, :orbital_period])
    filter!(row -> all(x -> x > 0, [row.mass, row.radius, row.orbital_period]), df_clean)

    # Dla parametrów gwiazdowych (metaliczność, masa gwiazdy) robimy imputację medianą,
    # żeby nie tracić planet tylko dlatego, że nie zważono gwiazdy.
    for col in [:star_metallicity, :star_mass, :star_teff]
        vals = collect(skipmissing(df_clean[!, col]))
        med = isempty(vals) ? 0.0 : median(vals)
        df_clean[!, col] = coalesce.(df_clean[!, col], med)
    end

    n = nrow(df_clean)
    println("Liczba planet zakwalifikowanych do PCA: $n")

    print_header("2. OBLICZANIE PCA")

    # Logarytmujemy to co ma rozkład potęgowy (Masa, Promień, Okres, Temp, Masa Gwiazdy)
    # Metaliczność zostawiamy liniowo (bo może być ujemna i już jest logarytmem)

    X_raw = Matrix{Float64}(undef, 6, n)
    X_raw[1, :] = log10.(df_clean.mass)
    X_raw[2, :] = log10.(df_clean.radius)
    X_raw[3, :] = log10.(df_clean.orbital_period)
    X_raw[4, :] = df_clean.star_metallicity # Bez log
    X_raw[5, :] = log10.(df_clean.star_mass .+ 1e-9)
    X_raw[6, :] = log10.(df_clean.star_teff .+ 1e-9)

    Z_model = fit(ZScoreTransform, X_raw, dims=2)
    X_std = StatsBase.transform(Z_model, X_raw)

    pca_model = fit(PCA, X_std; maxoutdim=3)
    X_proj = MultivariateStats.transform(pca_model, X_std)

    vars = principalvars(pca_model) ./ tprincipalvar(pca_model) * 100
    println("Wyjaśniona wariancja:")
    println("  PC1: $(round(vars[1], digits=1))% (Główny czynnik różnicujący principal component)")
    println("  PC2: $(round(vars[2], digits=1))% (Drugi czynnik)")

    loadings = projection(pca_model)
    var_names = ["Log(Masa)", "Log(Promień)", "Log(Okres)", "Met. Gwiazdy", "Log(Masa Gw.)", "Log(Temp Gw.)"]

    cluster_model = kmeans(X_proj, 4)
    df_clean.cluster = assignments(cluster_model)

    print_header("3. GENEROWANIE WYKRESÓW")

    fig = Figure(size=(1300, 900), fontsize=18)

    ax_pca = Axis(fig[1, 1:2],
        title="PCA Biplot: Mapa Przestrzeni  PLANET",
        xlabel="PC1 ($(round(vars[1], digits=1))%)",
        ylabel="PC2 ($(round(vars[2], digits=1))%)"
    )

    scatter3d!(ax_pca, X_proj[1, :], X_proj[2, :],
        color=df_clean.cluster, colormap=:plasma,
        markersize=6, transparency=true, label="Planety"
    )

    scale_factor = 4.0
    for i in 1:length(var_names)
        u = loadings[i, 1] * scale_factor
        v = loadings[i, 2] * scale_factor
        arrows3d!(ax_pca, [0], [0], [u], [v], color=:red)
        text!(ax_pca, u * 1.5, v * 1.5, text=var_names[i], color=:green, fontsize=15, font=:bold)
    end

    ax_phys = Axis(fig[2, 1],
        title="Fizyka: Masa vs Promień juz niema logarytmwo",
        xlabel="Masa [M_Jup] (log)", ylabel="Promień [R_Jup] (log)",
        xscale=log10, yscale=log10
    )
    scatter3d!(ax_phys, df_clean.mass, df_clean.radius,
        color=df_clean.cluster, colormap=:plasma,
        markersize=6
    )

    ax_star = Axis(fig[2, 2],
        title="Geneza: Metaliczność Gwiazdy vs Masa Planety",
        xlabel="Metaliczność [Fe/H]", ylabel="Masa Planety [M_Jup] ",
    )
    scatter!(ax_star, df_clean.star_metallicity, df_clean.mass,
        color=(:black, 0.3), markersize=5
    )
    X_met = df_clean.star_metallicity
    Y_mass = log10.(df_clean.mass)
    valid_idx = isfinite.(X_met) .& isfinite.(Y_mass)
    if sum(valid_idx) > 10
        X_reg = [ones(sum(valid_idx)) X_met[valid_idx]]
        b = X_reg \ Y_mass[valid_idx] # Rozwiązanie MNK
        xs = range(minimum(X_met), maximum(X_met), length=100)
        ys = b[1] .+ b[2] .* xs
        lines!(ax_star, xs, 10 .^ ys, color=:red, linewidth=3, label="Trend")
        axislegend(ax_star)
    end

    display(fig)
    println("Gotowe. Skibidi skibidi punkty ministerialne lecoooo.")

    println("\n=== INTERPRETACJA PCA ===")
    println("Spójrz na czerwone strzałki na wykresie PCA:")
    println("1. Jeśli strzałki 'Masa' i 'Promień' idą w tym samym kierunku (zazwyczaj wzdłuż PC1),")
    println("   oznacza to, że PC1 reprezentuje 'Rozmiar planety'.")
    println("2. Jeśli strzałka 'Metaliczność' jest prostopadła do masy, oznacza brak korelacji.")
    println("   Jeśli kąt jest ostry (<90 stopni), istnieje pozytywna korelacja.")
    println("3. Klastry (kolory) pokazują naturalne grupy planet wykryte przez algorytm.")
end

main()
