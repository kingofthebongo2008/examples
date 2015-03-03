namespace ColorYourWay

open WebSharper.Sitelets

[<Sealed>]
type Website() =
    interface IWebsite<Action> with
        member this.Sitelet = Site.Main
        member this.Actions = [Home; DigitalPapers]

type Global() =
    inherit System.Web.HttpApplication()

    member g.Application_Start(sender: obj, args: System.EventArgs) =
        ()

[<assembly: Website(typeof<Website>)>]
do ()
