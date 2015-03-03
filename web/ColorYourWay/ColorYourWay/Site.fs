namespace ColorYourWay

open WebSharper.Html.Server
open WebSharper.Sitelets

module Site =

    let makePageTitle ( s: string ) : string = 
        "Color Your Way" + " " + s

    let makePage title = 
        let t = makePageTitle title
        Skin.WithTemplate t <| fun ctx ->
            [
                Div [Text title]
            ]

    let homePage =
        let title = makePageTitle ""
        Skin.WithTemplate title <| fun ctx ->
            [
                Div [Text "HOME"]
            ]

    let digitalPapersPage = makePage "Digital Papers"
    let clipArtPage = makePage "Clip Art"
    let digitalStampsPage = makePage "Digital Stamps"
    let alphabetsPage = makePage "Alpha Bets"
    let printablesPage = makePage "Printables"
    let discountsPage = makePage "Discounts"
    let freebiesPage = makePage "Freebies"

    let Main =
        Sitelet.Sum [
            Sitelet.Content "/" Home homePage
            Sitelet.Content "/Digital Papers" DigitalPapers digitalPapersPage
            Sitelet.Content "/Clipart" ClipArt clipArtPage
            Sitelet.Content "/Digital Stamps" DigitalStamps digitalStampsPage
            Sitelet.Content "/Alphabets"  Alphabets alphabetsPage
            Sitelet.Content "/Printables" Printables printablesPage
            Sitelet.Content "/Discounts"  Discounts discountsPage
            Sitelet.Content "/Freebies"   Freebies freebiesPage
        ]

