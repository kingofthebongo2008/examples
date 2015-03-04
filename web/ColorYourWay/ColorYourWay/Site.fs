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

    let ( => ) text url =
        A [HRef url] -< [Text text]

    let makeHeader ctx =
        [
            
                    UL [Class "nav nav-pills"] -<
                    [
                        LI [ "Digital Papers" => ctx.Link DigitalPapers ]
                        LI [ "Clip Art"       => ctx.Link ClipArt       ]
                        LI [ "Digital Stamps" => ctx.Link DigitalStamps ]
                        LI [ "Alphabets"      => ctx.Link Alphabets     ]
                        LI [ "Printables"     => ctx.Link Printables    ]
                        LI [ "Discounts"      => ctx.Link Discounts     ]
                        LI [ "Freebies"       => ctx.Link Freebies      ]
                    ]
        ]

    let homeBody ctx =
        let r = [
                    Div [Text "HOME"]
                ]
        r

    let combine ( f :  Context<Action> -> Element list )   ( g :  Context<Action> -> Element list )  ( ctx : Context<Action> ) : Element list  =
        let a = f ctx
        let b = g ctx
        List.concat [ a ; b ]

    let homePage =
        let title = makePageTitle ""
        let header = makeHeader
        let f = combine makeHeader homeBody
        Skin.WithTemplate title f


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
            content "" Home homePage
            content "Digital Papers" DigitalPapers digitalPapersPage
            content "Clipart" ClipArt clipArtPage
            content "Digital Stamps" DigitalStamps digitalStampsPage
            content "Alphabets"  Alphabets alphabetsPage
            content "Printables" Printables printablesPage
            content "Discounts"  Discounts discountsPage
            content "Freebies"   Freebies freebiesPage
        ]

