module ColorYourWay.Dependencies

open WebSharper.Core.Resources;
open WebSharper.Core.Attributes;
open WebSharper.JQuery;

//<!-- Latest compiled and minified CSS -->
//<link rel="stylesheet" href="//maxcdn.bootstrapcdn.com/bootstrap/3.3.2/css/bootstrap.min.css"/>

//<!-- Optional theme -->
//<link rel="stylesheet" href="//maxcdn.bootstrapcdn.com/bootstrap/3.3.2/css/bootstrap-theme.min.css"/>

//<!-- Latest compiled and minified JavaScript -->
//<script src="//maxcdn.bootstrapcdn.com/bootstrap/3.3.2/js/bootstrap.min.js"></script>


[<Require(typeof<Resources.JQuery>)>]
[<Sealed>]
type Bootstrap() =
    inherit BaseResource( "//maxcdn.bootstrapcdn.com/bootstrap/3.3.2/", "css/bootstrap.min.css", "css/bootstrap-theme.min.css", "js/bootstrap.min.js")

[<assembly: Require(typeof<Bootstrap>)>]
do ()


