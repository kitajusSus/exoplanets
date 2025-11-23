# witam serdecznie julia analiza exoplanety pca


JAK ZAINSTALOWAC JULIA  NA MAC
[link](https://julialang.org/install/)


```bash
curl -fsSL https://install.julialang.org | sh
```

jak juz mamy julia zainstalowany i dodany do path to:


```bash
$ julia

## wtedy pojawi się ikonka z logiem i nazwa to coś to jest repl
# jeśli jestesmy w folderze z projektem to świetnie bo można wtedy włączyć środowisko

$ ]

## wtedy zmienia się lub powinno się zmieniać na coś w stylu (1.12) pkg>
# i jak juz mamy te cośtam to
pkg>
pkg> activate .
pkg> instantiate
# by z tego wyjsc trzeba kliknąć backspace

# I POWINNO WYGENEROWAĆ pobrać wszystkie potrzebne pakiety
# jak juz wszystko się pobierze to calecam by użyć

julia> using Revise
julia> includet("nik1.jl")
# i wszystko powinno się uruchomić samodzielnie potem jak coś zmieni uzytkownik
# to prosze wpisać
julia> main()




```


w kodzie jest miejsce na wpisywanie nazwy pików do analizy prosze pamietac
ze dostosowane jest to tylko do .csv i nazwy kolumn musza się zgadzać inaczej to
niczego nie policzy
