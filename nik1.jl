using CSV, DataFrames, Statistics, GLMakie, Clustering, LinearAlgebra, MultivariateStats, StatsBase, Printf, Random

function print_header(text)
    println("\n" * repeat("=", 60))
    println("=== $text ===")
    println(repeat("=", 60))
end

function prep_matrix(df, cols_log, cols_linear)
    n = nrow(df)
    n_vars = length(cols_log) + length(cols_linear)
    X = Matrix{Float64}(undef, n_vars, n)

    row_idx = 1
    for col in cols_log
        X[row_idx, :] = log10.(df[!, col] .+ 1e-9)
        row_idx += 1
    end
    for col in cols_linear
        X[row_idx, :] = df[!, col]
        row_idx += 1
    end
    return X
end
"""
FILENAME WPISUJEMY NAZWĘ KATALOGU Z DANYMI

"""
function main()
    print_header("1. WCZYTYWANIE I PRZYGOTOWANIE DANYCH")
    filename = "exoplanet.eu_catalog_all.csv"
    if !isfile(filename)
        if isfile("exoplanet.eu_catalog.csv")
            filename = "exoplanet.eu_catalog.csv"
        elseif isfile("exoplanet.eu_catalog_20-11-25_14_48_51.csv")
            filename = "exoplanet.eu_catalog_20-11-25_14_48_51.csv"
        else
            error("Brak pliku CSV! Sprawdź nazwę pliku.")
        end
    end

    df = CSV.read(filename, DataFrame)
    ####### NAZWY KOLUMN Z PLIKU CSV
    needed_cols = [:mass, :radius, :orbital_period, :star_metallicity, :star_mass, :star_teff, :planet_status, :name]

    valid_cols = intersect(needed_cols, propertynames(df))
    select!(df, valid_cols)

    for col in [:star_metallicity, :star_mass, :star_teff]
        if col in propertynames(df)
            vals = collect(skipmissing(df[!, col]))
            med = isempty(vals) ? 0.0 : median(vals)
            df[!, col] = coalesce.(df[!, col], med)
        end
    end

    m = passmissing(Float64).(df.mass)
    r = passmissing(Float64).(df.radius)
    df.density = m ./ (r .^ 3)

    # Usuwamy rekordy bez kluczowych danych do PCA
    dropmissing!(df, [:mass, :radius, :orbital_period])
    filter!(r -> r.mass > 0 && r.radius > 0 && r.orbital_period > 0, df)

    df_conf = filter(r -> coalesce(r.planet_status, "") == "Confirmed", df)
    df_cand = filter(r -> coalesce(r.planet_status, "") != "Confirmed", df)

    println("Liczba planet POTWIERDZONYCH (Baza): $(nrow(df_conf))")
    println("Liczba KANDYDATÓW (Do sprawdzenia): $(nrow(df_cand))")

    if nrow(df_cand) == 0
        println("UWAGA: Brak kandydatów. Kod wyświetli tylko potwierdzone.")
    end

    print_header("2. TRENOWANIE PCA (NA CONFIRMED)")

    cols_to_log = [:mass, :radius, :orbital_period, :star_mass, :star_teff]
    cols_linear = [:star_metallicity]

    X_train = prep_matrix(df_conf, cols_to_log, cols_linear)

    Z_model = fit(ZScoreTransform, X_train, dims=2)
    X_train_std = StatsBase.transform(Z_model, X_train)

    # 3. PCA
    pca_model = fit(PCA, X_train_std; maxoutdim=3)

    X_conf_proj = MultivariateStats.transform(pca_model, X_train_std)

    X_cand_proj = Matrix{Float64}(undef, 3, 0)
    if nrow(df_cand) > 0
        X_test = prep_matrix(df_cand, cols_to_log, cols_linear)
        X_test_std = StatsBase.transform(Z_model, X_test)
        X_cand_proj = MultivariateStats.transform(pca_model, X_test_std)
    end

    k = 4
    cluster_model = kmeans(X_conf_proj, k)
    df_conf.cluster = assignments(cluster_model)

    if nrow(df_cand) > 0
        centroids = cluster_model.centers
        function predict_cluster(point, centers)
            dists = [sum(abs2, point .- centers[:, i]) for i in 1:size(centers, 2)]
            return argmin(dists)
        end
        df_cand.cluster = [predict_cluster(X_cand_proj[:, i], centroids) for i in 1:nrow(df_cand)]
    end

    print_header("3. GENEROWANIE WYKRESÓW PORÓWNAWCZYCH")

    fig = Figure(size=(1600, 900), fontsize=16)

    vars = principalvars(pca_model) ./ tprincipalvar(pca_model) * 100
    pca_labels = ["PC$i ($(round(v, digits=1))%)" for (i, v) in enumerate(vars)]

    ax_conf = Axis3(fig[1, 1], title="BAZA: Potwierdzone",
        xlabel=pca_labels[1], ylabel=pca_labels[2], zlabel=pca_labels[3], azimuth=0.4)

    ax_cand = Axis3(fig[1, 2], title="TEST: Kandydaci",
        xlabel=pca_labels[1], ylabel=pca_labels[2], zlabel=pca_labels[3], azimuth=0.4)

    ax_p1 = Axis3(fig[2, 1], title="Fizyka: Potwierdzone",
        xlabel="Log(Masa)", ylabel="Log(Promień)", zlabel="Log(Gęstość)", azimuth=0.4)

    ax_p2 = Axis3(fig[2, 2], title="Fizyka: Kandydaci",
        xlabel="Log(Masa)", ylabel="Log(Promień)", zlabel="Log(Gęstość)", azimuth=0.4)


    sc_conf = scatter!(ax_conf, X_conf_proj[1, :], X_conf_proj[2, :], X_conf_proj[3, :],
        color=df_conf.cluster, colormap=:plasma, markersize=5, transparency=true)

    sc_conf.inspector_label = (self, i, p) -> begin
        return "Planeta: $(df_conf.name[i])\nTyp: Confirmed\nKlaster: $(df_conf.cluster[i])"
    end

    scatter!(ax_cand, X_conf_proj[1, :], X_conf_proj[2, :], X_conf_proj[3, :],
        color=(:gray, 0.1), markersize=3, inspectable=false)

    if nrow(df_cand) > 0
        sc_cand = scatter!(ax_cand, X_cand_proj[1, :], X_cand_proj[2, :], X_cand_proj[3, :],
            color=df_cand.cluster, colormap=:plasma, markersize=8, strokewidth=1, strokecolor=:black)

        sc_cand.inspector_label = (self, i, p) -> begin
            return "KANDYDAT: $(df_cand.name[i])\nStatus: $(df_cand.planet_status[i])\nKlaster: $(df_cand.cluster[i])"
        end
    end

    sc_p1 = scatter!(ax_p1,
        log10.(df_conf.mass),
        log10.(df_conf.radius),
        log10.(df_conf.density),
        color=df_conf.cluster, colormap=:plasma, markersize=6, transparency=true)

    sc_p1.inspector_label = (self, i, p) -> "Planeta: $(df_conf.name[i])"

    scatter!(ax_p2,
        log10.(df_conf.mass),
        log10.(df_conf.radius),
        log10.(df_conf.density),
        color=(:gray, 0.1), markersize=3, inspectable=false)

    if nrow(df_cand) > 0
        sc_p2 = scatter!(ax_p2,
            log10.(df_cand.mass),
            log10.(df_cand.radius),
            log10.(df_cand.density),
            color=df_cand.cluster, colormap=:plasma, markersize=8, strokewidth=1)

        sc_p2.inspector_label = (self, i, p) -> "Kandydat: $(df_cand.name[i])"
    end

    loadings = projection(pca_model)
    var_names = ["Masa", "Promień", "Okres", "Masa Gw", "Temp Gw", "Met. Gw"]
    scale = 6.0
    origins = [Point3f(0, 0, 0) for _ in 1:6]
    dirs = [Point3f(loadings[i, 1], loadings[i, 2], loadings[i, 3]) * scale for i in 1:6]

    for ax in [ax_conf, ax_cand]
        arrows!(ax, origins, dirs, color=:red, linewidth=0.01, arrowsize=0.02)
        for i in 1:6
            text!(ax, dirs[i] * 1.1, text=var_names[i], color=:red, fontsize=15, font=:bold)
        end
    end

    DataInspector(fig)

    display(fig)
    println("Gotowe. Wykresy 3D (Logarytmy obliczone ręcznie).")

    println("\n=== NAJBARDZIEJ ODSTAJĄCY KANDYDACI (Anomalie PCA) ===")
    if nrow(df_cand) > 0
        dists = vec(sum(X_cand_proj .^ 2, dims=1))
        perm = sortperm(dists, rev=true)

        println("Top 5 anomalii (najdziwniejsze parametry):")
        for i in 1:min(5, length(perm))
            idx = perm[i]
            name = df_cand.name[idx]
            dist = round(sqrt(dists[idx]), digits=2)
            status = df_cand.planet_status[idx]
            println("$i. $name (Odległość PCA: $dist) - Status: $status")
        end
    end
end

main()

