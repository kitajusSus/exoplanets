using CSV, DataFrames, Statistics, GLMakie, Clustering, LinearAlgebra, MultivariateStats, StatsBase, Printf, Random

# ==============================================================================
# 1. FUNKCJE POMOCNICZE
# ==============================================================================

function print_header(text)
    println("\n" * repeat("=", 60))
    println("=== $text ===")
    println(repeat("=", 60))
end

"""
Oblicza wskaźniki barwy (Color Indices) na podstawie surowych magnitudo.
Dla Twojego pliku SDSS kolumny to: u, g, r, i, z.
"""
function prep_features(df)
    # Sprawdzenie czy kolumny istnieją
    required_cols = [:u, :g, :r, :i, :z]
    for col in required_cols
        if !(col in propertynames(df))
            error("Brak kolumny w pliku CSV: $col. Sprawdź nagłówki.")
        end
    end

    # Obliczamy kolory (różnice jasności)
    # To jest standard w astronomii: u-g, g-r, r-i, i-z
    c1 = df.u .- df.g
    c2 = df.g .- df.r
    c3 = df.r .- df.i
    c4 = df.i .- df.z

    # Tworzymy macierz cech (Wiersze = Cechy, Kolumny = Obiekty)
    # Dodajemy też jasność 'r', bo czasem jasne obiekty to częściej gwiazdy
    # Ale w czystej analizie kolorów wystarczą c1..c4.
    X = Matrix(hcat(c1, c2, c3, c4)')
    return Float64.(X)
end


function main()
    print_header("1. WCZYTYWANIE DANYCH")

    filename = "Skyserver_SQL2_27_2018 6_51_39 PM.csv"

    if !isfile(filename)
        println("BŁĄD: Nie znaleziono pliku '$filename'.")
        return
    end

    println("Wczytywanie: $filename...")
    df = CSV.read(filename, DataFrame)

    # Filtrowanie błędnych danych
    filter!(row -> all(x -> x > -100 && x < 100, [row.u, row.g, row.r, row.i, row.z]), df)

    X_raw = prep_features(df)

    Z_model = fit(ZScoreTransform, X_raw, dims=2)
    X_std = StatsBase.transform(Z_model, X_raw)

    print_header("2. PCA (REDUKCJA WYMIAROWOŚCI)")

    pca_model = fit(PCA, X_std; maxoutdim=3)
    X_proj = MultivariateStats.transform(pca_model, X_std)

    expl_var = principalvars(pca_model) ./ tprincipalvar(pca_model) * 100
    println("Wariancja wyjaśniona:")
    for (i, v) in enumerate(expl_var)
        @printf "PC%d: %.2f%%\n" i v
    end

    print_header("3. KLASTERYZACJA (K-MEANS)")

    k = 3
    cluster_model = kmeans(X_proj, k)
    df.cluster = assignments(cluster_model)

    println("\nMacierz pomyłek (Cluster vs Klasa):")
    for i in 1:k
        sub = filter(r -> r.cluster == i, df)
        counts = countmap(sub.class)
        s = get(counts, "STAR", 0)
        q = get(counts, "QSO", 0)
        g = get(counts, "GALAXY", 0)
        println("Cluster $i => STAR: $s, QSO: $q, GALAXY: $g")
    end

    print_header("4. WIZUALIZACJA (GLMakie)")

    fig = Figure(size=(1600, 900), fontsize=20)

    function get_color(cls)
        if cls == "STAR"
            return :blue
        elseif cls == "QSO"
            return :red
        elseif cls == "GALAXY"
            return :green
        else
            return :gray
        end
    end

    # Dla wykresu 3D ustawiamy kolory z przezroczystością (alfa = 0.5)
    pt_colors_3d = [(get_color(c), 0.5) for c in df.class]

    # --- WYKRES 1: PCA (3D) ---
    ax_pca = Axis3(fig[1, 1],
        title="Przestrzeń Cech PCA (3D)",
        xlabel="PC1", ylabel="PC2", zlabel="PC3", azimuth=0.7)

    scatter!(ax_pca, X_proj[1, :], X_proj[2, :], X_proj[3, :],
        color=pt_colors_3d, markersize=4, transparency=true)

    # --- WYKRES 2: DIAGRAM KOLOR-KOLOR (2D) ---
    ax_color = Axis(fig[1, 2],
        title="Diagram Kolor-Kolor (u-g vs g-r)",
        xlabel="u - g (UV)", ylabel="g - r (Wizualny)")

    ug = df.u .- df.g
    gr = df.g .- df.r

    for cls in ["STAR", "QSO", "GALAXY"]
        mask = df.class .== cls
        if sum(mask) > 0
            # POPRAWKA: Przezroczystość w kolorze, transparency=true jako flaga
            c_sym = get_color(cls)
            scatter!(ax_color, ug[mask], gr[mask],
                color=(c_sym, 0.6),  # Tu ustawiamy 60% widoczności
                markersize=6, label=cls, transparency=true)
        end
    end

    axislegend(ax_color, position=:rb)

    # --- WYKRES 3: ANOMALIE ---
    stars_mask = df.class .== "STAR"
    if count(stars_mask) > 0
        center_star = vec(mean(X_proj[:, stars_mask], dims=2))
        dists = [norm(X_proj[:, i] .- center_star) for i in 1:nrow(df)]
        df.dist_from_stars = dists

        ax_hist = Axis(fig[2, 1:2],
            title="Histogram odległości od centrum Gwiazd",
            xlabel="Odległość PCA", ylabel="Liczba obiektów")

        hist!(ax_hist, df.dist_from_stars[df.class.=="STAR"], color=(:blue, 0.5), label="Gwiazdy", bins=50)
        hist!(ax_hist, df.dist_from_stars[df.class.=="QSO"], color=(:red, 0.5), label="Kwazary", bins=50)
        axislegend(ax_hist)

        # Wypisanie kandydatów
        sort!(df, :dist_from_stars, rev=true)
        println("\nTop 5 anomalii (kandydaci):")
        println(first(select(df, [:objid, :class, :u, :g, :dist_from_stars]), 5))
    end

    display(fig)
    println("Gotowe.")
end

main()
