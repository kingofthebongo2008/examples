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

    let encodeSectionUrl (s : string ) =
        let r = s.Replace(' ','-')
        r
    let encodeSection (name : string) =
        let e = encodeSectionUrl name;
        let r = "/" + e;
        r

    let content name action page = 
        let e = encodeSection name
        Sitelet.Content e action page

    let Main =
        Sitelet.Sum [
            content "/" Home homePage
            content "Digital Papers" DigitalPapers digitalPapersPage
            content "Clipart" ClipArt clipArtPage
            content "Digital Stamps" DigitalStamps digitalStampsPage
            content "Alphabets"  Alphabets alphabetsPage
            content "Printables" Printables printablesPage
            content "Discounts"  Discounts discountsPage
            content "Freebies"   Freebies freebiesPage
        ]

