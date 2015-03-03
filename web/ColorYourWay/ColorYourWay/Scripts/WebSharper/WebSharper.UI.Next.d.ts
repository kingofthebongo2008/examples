declare module WebSharper {
    module UI {
        module Next {
            module Var {
                var Create : {
                    <_M1>(v: _M1): __ABBREV.__Next.Var<_M1>;
                };
                var Get : {
                    <_M1>(_var: __ABBREV.__Next.Var<_M1>): _M1;
                };
                var Set : {
                    <_M1>(_var: __ABBREV.__Next.Var<_M1>, value: _M1): void;
                };
            }
            module Var1 {
                var SetFinal : {
                    <_M1>(_var: __ABBREV.__Next.Var<_M1>, value: _M1): void;
                };
                var Update : {
                    <_M1>(_var: __ABBREV.__Next.Var<_M1>, fn: {
                        (x: _M1): _M1;
                    }): void;
                };
                var GetId : {
                    <_M1>(_var: __ABBREV.__Next.Var<_M1>): number;
                };
                var Observe : {
                    <_M1>(_var: __ABBREV.__Next.Var<_M1>): any;
                };
            }
            module View {
                var FromVar : {
                    <_M1>(_var: __ABBREV.__Next.Var<_M1>): __ABBREV.__Next.View<_M1>;
                };
                var CreateLazy : {
                    <_M1>(observe: {
                        (): any;
                    }): __ABBREV.__Next.View<_M1>;
                };
                var Map : {
                    <_M1, _M2>(fn: {
                        (x: _M1): _M2;
                    }, _arg1: __ABBREV.__Next.View<_M1>): __ABBREV.__Next.View<_M2>;
                };
                var CreateLazy2 : {
                    <_M1, _M2, _M3>(snapFn: {
                        (x: any): {
                            (x: any): any;
                        };
                    }, _arg3: __ABBREV.__Next.View<_M1>, _arg2: __ABBREV.__Next.View<_M2>): __ABBREV.__Next.View<_M3>;
                };
                var Map2 : {
                    <_M1, _M2, _M3>(fn: {
                        (x: _M1): {
                            (x: _M2): _M3;
                        };
                    }, v1: __ABBREV.__Next.View<_M1>, v2: __ABBREV.__Next.View<_M2>): __ABBREV.__Next.View<_M3>;
                };
                var MapAsync : {
                    <_M1, _M2>(fn: {
                        (x: _M1): any;
                    }, _arg4: __ABBREV.__Next.View<_M1>): __ABBREV.__Next.View<_M2>;
                };
                var SnapshotOn : {
                    <_M1, _M2>(def: _M1, _arg6: __ABBREV.__Next.View<_M2>, _arg5: __ABBREV.__Next.View<_M1>): __ABBREV.__Next.View<_M1>;
                };
                var UpdateWhile : {
                    <_M1>(def: _M1, v1: __ABBREV.__Next.View<boolean>, v2: __ABBREV.__Next.View<_M1>): __ABBREV.__Next.View<_M1>;
                };
                var ConvertBy : {
                    <_M1, _M2, _M3>(key: {
                        (x: _M1): _M3;
                    }, conv: {
                        (x: _M1): _M2;
                    }, view: __ABBREV.__Next.View<__ABBREV.__WebSharper.seq<_M1>>): __ABBREV.__Next.View<__ABBREV.__WebSharper.seq<_M2>>;
                };
                var Convert : {
                    <_M1, _M2>(conv: {
                        (x: _M1): _M2;
                    }, view: __ABBREV.__Next.View<__ABBREV.__WebSharper.seq<_M1>>): __ABBREV.__Next.View<__ABBREV.__WebSharper.seq<_M2>>;
                };
                var ConvertSeqNode : {
                    <_M1, _M2>(conv: {
                        (x: __ABBREV.__Next.View<_M1>): _M2;
                    }, value: _M1): any;
                };
                var ConvertSeqBy : {
                    <_M1, _M2, _M3>(key: {
                        (x: _M1): _M3;
                    }, conv: {
                        (x: __ABBREV.__Next.View<_M1>): _M2;
                    }, view: __ABBREV.__Next.View<__ABBREV.__WebSharper.seq<_M1>>): __ABBREV.__Next.View<__ABBREV.__WebSharper.seq<_M2>>;
                };
                var get_Do : {
                    (): __ABBREV.__Next.ViewBuilder;
                };
            }
            module View1 {
                var ConvertSeq : {
                    <_M1, _M2>(conv: {
                        (x: __ABBREV.__Next.View<_M1>): _M2;
                    }, view: __ABBREV.__Next.View<__ABBREV.__WebSharper.seq<_M1>>): __ABBREV.__Next.View<__ABBREV.__WebSharper.seq<_M2>>;
                };
                var Join : {
                    <_M1>(_arg7: __ABBREV.__Next.View<__ABBREV.__Next.View<_M1>>): __ABBREV.__Next.View<_M1>;
                };
                var Bind : {
                    <_M1, _M2>(fn: {
                        (x: _M1): __ABBREV.__Next.View<_M2>;
                    }, view: __ABBREV.__Next.View<_M1>): __ABBREV.__Next.View<_M2>;
                };
                var Const : {
                    <_M1>(x: _M1): __ABBREV.__Next.View<_M1>;
                };
                var Sink : {
                    <_M1>(act: {
                        (x: _M1): void;
                    }, _arg8: __ABBREV.__Next.View<_M1>): void;
                };
                var Apply : {
                    <_M1, _M2>(fn: __ABBREV.__Next.View<{
                        (x: _M1): _M2;
                    }>, view: __ABBREV.__Next.View<_M1>): __ABBREV.__Next.View<_M2>;
                };
            }
            module Key {
                var Fresh : {
                    (): __ABBREV.__Next.Key;
                };
            }
            module Model {
                var Create : {
                    <_M1, _M2>(proj: {
                        (x: _M1): _M2;
                    }, init: _M1): __ABBREV.__Next.Model<_M2, _M1>;
                };
                var Update : {
                    <_M1, _M2>(update: {
                        (x: _M1): void;
                    }, _arg1: __ABBREV.__Next.Model<_M2, _M1>): void;
                };
                var View : {
                    <_M1, _M2>(_arg2: __ABBREV.__Next.Model<_M1, _M2>): __ABBREV.__Next.View<_M1>;
                };
            }
            module ListModel {
                var Create : {
                    <_M1, _M2>(key: {
                        (x: _M2): _M1;
                    }, init: __ABBREV.__WebSharper.seq<_M2>): __ABBREV.__Next.ListModel<_M1, _M2>;
                };
                var FromSeq : {
                    <_M1>(xs: __ABBREV.__WebSharper.seq<_M1>): __ABBREV.__Next.ListModel<_M1, _M1>;
                };
            }
            module ListModel1 {
                var View : {
                    <_M1, _M2>(m: __ABBREV.__Next.ListModel<_M1, _M2>): __ABBREV.__Next.View<__ABBREV.__WebSharper.seq<_M2>>;
                };
            }
            module Interpolation1 {
                var get_Double : {
                    (): __ABBREV.__Next.Interpolation<number>;
                };
            }
            module Easing {
                var Custom : {
                    (f: {
                        (x: number): number;
                    }): __ABBREV.__Next.Easing;
                };
                var get_CubicInOut : {
                    (): __ABBREV.__Next.Easing;
                };
            }
            module An {
                var Append : {
                    (_arg2: __ABBREV.__Next.An, _arg1: __ABBREV.__Next.An): __ABBREV.__Next.An;
                };
                var Concat : {
                    (xs: __ABBREV.__WebSharper.seq<__ABBREV.__Next.An>): __ABBREV.__Next.An;
                };
                var Const : {
                    <_M1>(v: _M1): any;
                };
                var Simple : {
                    <_M1>(inter: __ABBREV.__Next.Interpolation<_M1>, easing: __ABBREV.__Next.Easing, dur: number, startValue: _M1, endValue: _M1): any;
                };
                var Delayed : {
                    <_M1>(inter: __ABBREV.__Next.Interpolation<_M1>, easing: __ABBREV.__Next.Easing, dur: number, delay: number, startValue: _M1, endValue: _M1): any;
                };
                var Map : {
                    <_M1, _M2>(f: {
                        (x: _M1): _M2;
                    }, anim: any): any;
                };
                var Pack : {
                    (anim: any): __ABBREV.__Next.An;
                };
                var Play : {
                    (anim: __ABBREV.__Next.An): any;
                };
                var Run : {
                    (k: {
                        (): void;
                    }, anim: any): any;
                };
                var WhenDone : {
                    (f: {
                        (): void;
                    }, main: __ABBREV.__Next.An): __ABBREV.__Next.An;
                };
                var get_Empty : {
                    (): __ABBREV.__Next.An;
                };
            }
            module Trans {
                var AnimateChange : {
                    <_M1>(tr: any, x: _M1, y: _M1): any;
                };
                var AnimateEnter : {
                    <_M1>(tr: any, x: _M1): any;
                };
                var AnimateExit : {
                    <_M1>(tr: any, x: _M1): any;
                };
                var CanAnimateChange : {
                    <_M1>(tr: any): boolean;
                };
                var CanAnimateEnter : {
                    <_M1>(tr: any): boolean;
                };
                var CanAnimateExit : {
                    <_M1>(tr: any): boolean;
                };
                var Trivial : {
                    <_M1>(): any;
                };
                var Create : {
                    <_M1>(ch: {
                        (x: _M1): {
                            (x: _M1): any;
                        };
                    }): any;
                };
                var Change : {
                    <_M1>(ch: {
                        (x: _M1): {
                            (x: _M1): any;
                        };
                    }, tr: any): any;
                };
                var Enter : {
                    <_M1>(f: {
                        (x: _M1): any;
                    }, tr: any): any;
                };
                var Exit : {
                    <_M1>(f: {
                        (x: _M1): any;
                    }, tr: any): any;
                };
            }
            module Attr {
                var Animated : {
                    <_M1>(name: string, tr: any, view: __ABBREV.__Next.View<_M1>, value: {
                        (x: _M1): string;
                    }): __ABBREV.__Next.Attr;
                };
                var AnimatedStyle : {
                    <_M1>(name: string, tr: any, view: __ABBREV.__Next.View<_M1>, value: {
                        (x: _M1): string;
                    }): __ABBREV.__Next.Attr;
                };
                var Dynamic : {
                    (name: string, value: __ABBREV.__Next.View<string>): __ABBREV.__Next.Attr;
                };
                var DynamicCustom : {
                    <_M1>(set: {
                        (x: __ABBREV.__Dom.Element): {
                            (x: _M1): void;
                        };
                    }, value: __ABBREV.__Next.View<_M1>): __ABBREV.__Next.Attr;
                };
                var DynamicStyle : {
                    (name: string, value: __ABBREV.__Next.View<string>): __ABBREV.__Next.Attr;
                };
                var Create : {
                    (name: string, value: string): __ABBREV.__Next.Attr;
                };
                var Style : {
                    (name: string, value: string): __ABBREV.__Next.Attr;
                };
                var Handler : {
                    (name: string, callback: {
                        (x: __ABBREV.__Dom.Event): void;
                    }): __ABBREV.__Next.Attr;
                };
                var Class : {
                    (name: string): __ABBREV.__Next.Attr;
                };
                var DynamicClass : {
                    <_M1>(name: string, view: __ABBREV.__Next.View<_M1>, apply: {
                        (x: _M1): boolean;
                    }): __ABBREV.__Next.Attr;
                };
                var DynamicPred : {
                    (name: string, predView: __ABBREV.__Next.View<boolean>, valView: __ABBREV.__Next.View<string>): __ABBREV.__Next.Attr;
                };
                var Append : {
                    (a: __ABBREV.__Next.Attr, b: __ABBREV.__Next.Attr): __ABBREV.__Next.Attr;
                };
                var Concat : {
                    (xs: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>): __ABBREV.__Next.Attr;
                };
                var get_Empty : {
                    (): __ABBREV.__Next.Attr;
                };
            }
            module Doc {
                var Append : {
                    (a: __ABBREV.__Next.Doc, b: __ABBREV.__Next.Doc): __ABBREV.__Next.Doc;
                };
                var Concat : {
                    (xs: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var Elem : {
                    (name: __ABBREV.__Dom.Element, attr: __ABBREV.__Next.Attr, children: __ABBREV.__Next.Doc): __ABBREV.__Next.Doc;
                };
                var Element : {
                    (name: string, attr: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, children: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var SvgElement : {
                    (name: string, attr: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, children: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var Static : {
                    (el: __ABBREV.__Dom.Element): __ABBREV.__Next.Doc;
                };
                var EmbedView : {
                    (view: __ABBREV.__Next.View<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var TextNode : {
                    (v: string): __ABBREV.__Next.Doc;
                };
                var TextView : {
                    (txt: __ABBREV.__Next.View<string>): __ABBREV.__Next.Doc;
                };
                var Run : {
                    (parent: __ABBREV.__Dom.Element, doc: __ABBREV.__Next.Doc): void;
                };
                var RunById : {
                    (id: string, tr: __ABBREV.__Next.Doc): void;
                };
                var AsPagelet : {
                    (doc: __ABBREV.__Next.Doc): __ABBREV.__Client.Pagelet;
                };
                var Flatten : {
                    <_M1>(view: __ABBREV.__Next.View<_M1>): __ABBREV.__Next.Doc;
                };
                var Convert : {
                    <_M1>(render: {
                        (x: _M1): __ABBREV.__Next.Doc;
                    }, view: __ABBREV.__Next.View<__ABBREV.__WebSharper.seq<_M1>>): __ABBREV.__Next.Doc;
                };
                var ConvertBy : {
                    <_M1, _M2>(key: {
                        (x: _M1): _M2;
                    }, render: {
                        (x: _M1): __ABBREV.__Next.Doc;
                    }, view: __ABBREV.__Next.View<__ABBREV.__WebSharper.seq<_M1>>): __ABBREV.__Next.Doc;
                };
                var ConvertSeq : {
                    <_M1>(render: {
                        (x: __ABBREV.__Next.View<_M1>): __ABBREV.__Next.Doc;
                    }, view: __ABBREV.__Next.View<__ABBREV.__WebSharper.seq<_M1>>): __ABBREV.__Next.Doc;
                };
                var ConvertSeqBy : {
                    <_M1, _M2>(key: {
                        (x: _M1): _M2;
                    }, render: {
                        (x: __ABBREV.__Next.View<_M1>): __ABBREV.__Next.Doc;
                    }, view: __ABBREV.__Next.View<__ABBREV.__WebSharper.seq<_M1>>): __ABBREV.__Next.Doc;
                };
                var InputInternal : {
                    (attr: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, _var: __ABBREV.__Next.Var<string>, inputTy: __ABBREV.__Next.InputControlType): __ABBREV.__Next.Doc;
                };
                var Input : {
                    (attr: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, _var: __ABBREV.__Next.Var<string>): __ABBREV.__Next.Doc;
                };
                var PasswordBox : {
                    (attr: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, _var: __ABBREV.__Next.Var<string>): __ABBREV.__Next.Doc;
                };
                var InputArea : {
                    (attr: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, _var: __ABBREV.__Next.Var<string>): __ABBREV.__Next.Doc;
                };
                var Select : {
                    <_M1>(attrs: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, show: {
                        (x: _M1): string;
                    }, options: __ABBREV.__List.T<_M1>, current: __ABBREV.__Next.Var<_M1>): __ABBREV.__Next.Doc;
                };
                var CheckBox : {
                    <_M1>(attrs: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, item: _M1, chk: __ABBREV.__Next.Var<__ABBREV.__List.T<_M1>>): __ABBREV.__Next.Doc;
                };
                var Clickable : {
                    (elem: string, action: {
                        (): void;
                    }): __ABBREV.__Dom.Element;
                };
                var Button : {
                    (caption: string, attrs: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, action: {
                        (): void;
                    }): __ABBREV.__Next.Doc;
                };
                var Link : {
                    (caption: string, attrs: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, action: {
                        (): void;
                    }): __ABBREV.__Next.Doc;
                };
                var Radio : {
                    <_M1>(attrs: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, value: _M1, _var: __ABBREV.__Next.Var<_M1>): __ABBREV.__Next.Doc;
                };
                var get_Empty : {
                    (): __ABBREV.__Next.Doc;
                };
            }
            module Flow1 {
                var Map : {
                    <_M1, _M2>(f: {
                        (x: _M1): _M2;
                    }, x: any): any;
                };
                var Bind : {
                    <_M1, _M2>(m: any, k: {
                        (x: _M1): any;
                    }): any;
                };
                var Return : {
                    <_M1>(x: _M1): any;
                };
                var Embed : {
                    <_M1>(fl: any): __ABBREV.__Next.Doc;
                };
                var Define : {
                    <_M1>(f: {
                        (x: {
                            (x: _M1): void;
                        }): __ABBREV.__Next.Doc;
                    }): any;
                };
                var Static : {
                    (doc: __ABBREV.__Next.Doc): any;
                };
            }
            module Flow {
                var get_Do : {
                    (): __ABBREV.__Next.FlowBuilder;
                };
            }
            module RouteMap1 {
                var Create : {
                    <_M1>(ser: {
                        (x: _M1): __ABBREV.__List.T<string>;
                    }, des: {
                        (x: __ABBREV.__List.T<string>): _M1;
                    }): any;
                };
                var Install : {
                    <_M1>(map: any): __ABBREV.__Next.Var<_M1>;
                };
            }
            module Router {
                var Dir : {
                    <_M1>(prefix: string, sites: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Router<_M1>>): __ABBREV.__Next.Router<_M1>;
                };
                var Install : {
                    <_M1>(key: {
                        (x: _M1): __ABBREV.__Next.RouteId;
                    }, site: __ABBREV.__Next.Router<_M1>): __ABBREV.__Next.Var<_M1>;
                };
                var Merge : {
                    <_M1>(sites: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Router<_M1>>): __ABBREV.__Next.Router<_M1>;
                };
            }
            module Router1 {
                var Prefix : {
                    <_M1>(prefix: string, _arg1: __ABBREV.__Next.Router<_M1>): __ABBREV.__Next.Router<_M1>;
                };
                var Route : {
                    <_M1, _M2>(r: any, init: _M1, render: {
                        (x: __ABBREV.__Next.RouteId): {
                            (x: __ABBREV.__Next.Var<_M1>): _M2;
                        };
                    }): __ABBREV.__Next.Router<_M2>;
                };
            }
            module Input {
                module Mouse {
                    var get_Position : {
                        (): __ABBREV.__Next.View<any>;
                    };
                    var get_LeftPressed : {
                        (): __ABBREV.__Next.View<boolean>;
                    };
                    var get_MiddlePressed : {
                        (): __ABBREV.__Next.View<boolean>;
                    };
                    var get_RightPressed : {
                        (): __ABBREV.__Next.View<boolean>;
                    };
                    var get_MousePressed : {
                        (): __ABBREV.__Next.View<boolean>;
                    };
                }
                module Keyboard {
                    var IsPressed : {
                        (key: number): __ABBREV.__Next.View<boolean>;
                    };
                    var get_KeysPressed : {
                        (): __ABBREV.__Next.View<__ABBREV.__List.T<number>>;
                    };
                    var get_LastPressed : {
                        (): __ABBREV.__Next.View<number>;
                    };
                }
                interface Mouse {
                }
                interface Keyboard {
                }
                var MousePosSt1 : {
                    (): any;
                };
                var MouseBtnSt1 : {
                    (): any;
                };
                var ActivateButtonListener : {
                    (): void;
                };
                var KeyListenerState : {
                    (): any;
                };
                var ActivateKeyListener : {
                    (): void;
                };
            }
            module Html {
                module SvgAttributes {
                    var AccentHeight : {
                        (): string;
                    };
                    var Accumulate : {
                        (): string;
                    };
                    var Additive : {
                        (): string;
                    };
                    var AlignmentBaseline : {
                        (): string;
                    };
                    var Ascent : {
                        (): string;
                    };
                    var AttributeName : {
                        (): string;
                    };
                    var AttributeType : {
                        (): string;
                    };
                    var Azimuth : {
                        (): string;
                    };
                    var BaseFrequency : {
                        (): string;
                    };
                    var BaselineShift : {
                        (): string;
                    };
                    var Begin : {
                        (): string;
                    };
                    var Bias : {
                        (): string;
                    };
                    var CalcMode : {
                        (): string;
                    };
                    var Class : {
                        (): string;
                    };
                    var Clip : {
                        (): string;
                    };
                    var ClipPathUnits : {
                        (): string;
                    };
                    var ClipPath : {
                        (): string;
                    };
                    var ClipRule : {
                        (): string;
                    };
                    var Color : {
                        (): string;
                    };
                    var ColorInterpolation : {
                        (): string;
                    };
                    var ColorInterpolationFilters : {
                        (): string;
                    };
                    var ColorProfile : {
                        (): string;
                    };
                    var ColorRendering : {
                        (): string;
                    };
                    var ContentScriptType : {
                        (): string;
                    };
                    var ContentStyleType : {
                        (): string;
                    };
                    var Cursor : {
                        (): string;
                    };
                    var CX : {
                        (): string;
                    };
                    var CY : {
                        (): string;
                    };
                    var D : {
                        (): string;
                    };
                    var DiffuseConstant : {
                        (): string;
                    };
                    var Direction : {
                        (): string;
                    };
                    var Display : {
                        (): string;
                    };
                    var Divisor : {
                        (): string;
                    };
                    var DominantBaseline : {
                        (): string;
                    };
                    var Dur : {
                        (): string;
                    };
                    var DX : {
                        (): string;
                    };
                    var DY : {
                        (): string;
                    };
                    var EdgeMode : {
                        (): string;
                    };
                    var Elevation : {
                        (): string;
                    };
                    var End : {
                        (): string;
                    };
                    var ExternalResourcesRequired : {
                        (): string;
                    };
                    var Fill : {
                        (): string;
                    };
                    var FillOpacity : {
                        (): string;
                    };
                    var FillRule : {
                        (): string;
                    };
                    var Filter : {
                        (): string;
                    };
                    var FilterRes : {
                        (): string;
                    };
                    var FilterUnits : {
                        (): string;
                    };
                    var FloodColor : {
                        (): string;
                    };
                    var FloodOpacity : {
                        (): string;
                    };
                    var FontFamily : {
                        (): string;
                    };
                    var FontSize : {
                        (): string;
                    };
                    var FontSizeAdjust : {
                        (): string;
                    };
                    var FontStretch : {
                        (): string;
                    };
                    var FontStyle : {
                        (): string;
                    };
                    var FontVariant : {
                        (): string;
                    };
                    var FontWeight : {
                        (): string;
                    };
                    var From : {
                        (): string;
                    };
                    var GradientTransform : {
                        (): string;
                    };
                    var GradientUnits : {
                        (): string;
                    };
                    var Height : {
                        (): string;
                    };
                    var ImageRendering : {
                        (): string;
                    };
                    var IN : {
                        (): string;
                    };
                    var In2 : {
                        (): string;
                    };
                    var K1 : {
                        (): string;
                    };
                    var K2 : {
                        (): string;
                    };
                    var K3 : {
                        (): string;
                    };
                    var K4 : {
                        (): string;
                    };
                    var KernelMatrix : {
                        (): string;
                    };
                    var KernelUnitLength : {
                        (): string;
                    };
                    var Kerning : {
                        (): string;
                    };
                    var KeySplines : {
                        (): string;
                    };
                    var KeyTimes : {
                        (): string;
                    };
                    var LetterSpacing : {
                        (): string;
                    };
                    var LightingColor : {
                        (): string;
                    };
                    var LimitingConeAngle : {
                        (): string;
                    };
                    var Local : {
                        (): string;
                    };
                    var MarkerEnd : {
                        (): string;
                    };
                    var MarkerMid : {
                        (): string;
                    };
                    var MarkerStart : {
                        (): string;
                    };
                    var MarkerHeight : {
                        (): string;
                    };
                    var MarkerUnits : {
                        (): string;
                    };
                    var MarkerWidth : {
                        (): string;
                    };
                    var Mask : {
                        (): string;
                    };
                    var MaskContentUnits : {
                        (): string;
                    };
                    var MaskUnits : {
                        (): string;
                    };
                    var Max : {
                        (): string;
                    };
                    var Min : {
                        (): string;
                    };
                    var Mode : {
                        (): string;
                    };
                    var NumOctaves : {
                        (): string;
                    };
                    var Opacity : {
                        (): string;
                    };
                    var Operator : {
                        (): string;
                    };
                    var Order : {
                        (): string;
                    };
                    var Overflow : {
                        (): string;
                    };
                    var PaintOrder : {
                        (): string;
                    };
                    var PathLength : {
                        (): string;
                    };
                    var PatternContentUnits : {
                        (): string;
                    };
                    var PatternTransform : {
                        (): string;
                    };
                    var PatternUnits : {
                        (): string;
                    };
                    var PointerEvents : {
                        (): string;
                    };
                    var Points : {
                        (): string;
                    };
                    var PointsAtX : {
                        (): string;
                    };
                    var PointsAtY : {
                        (): string;
                    };
                    var PointsAtZ : {
                        (): string;
                    };
                    var PreserveAlpha : {
                        (): string;
                    };
                    var PreserveAspectRatio : {
                        (): string;
                    };
                    var PrimitiveUnits : {
                        (): string;
                    };
                    var R : {
                        (): string;
                    };
                    var Radius : {
                        (): string;
                    };
                    var RepeatCount : {
                        (): string;
                    };
                    var RepeatDur : {
                        (): string;
                    };
                    var RequiredFeatures : {
                        (): string;
                    };
                    var Restart : {
                        (): string;
                    };
                    var Result : {
                        (): string;
                    };
                    var RX : {
                        (): string;
                    };
                    var RY : {
                        (): string;
                    };
                    var Scale : {
                        (): string;
                    };
                    var Seed : {
                        (): string;
                    };
                    var ShapeRendering : {
                        (): string;
                    };
                    var SpecularConstant : {
                        (): string;
                    };
                    var SpecularExponent : {
                        (): string;
                    };
                    var StdDeviation : {
                        (): string;
                    };
                    var StitchTiles : {
                        (): string;
                    };
                    var StopColor : {
                        (): string;
                    };
                    var StopOpacity : {
                        (): string;
                    };
                    var Stroke : {
                        (): string;
                    };
                    var StrokeDashArray : {
                        (): string;
                    };
                    var StrokeDashOffset : {
                        (): string;
                    };
                    var StrokeLineCap : {
                        (): string;
                    };
                    var StrokeLineJoin : {
                        (): string;
                    };
                    var StrokeMiterLimit : {
                        (): string;
                    };
                    var StrokeOpacity : {
                        (): string;
                    };
                    var StrokeWidth : {
                        (): string;
                    };
                    var Style : {
                        (): string;
                    };
                    var SurfaceScale : {
                        (): string;
                    };
                    var TargetX : {
                        (): string;
                    };
                    var TargetY : {
                        (): string;
                    };
                    var TextAnchor : {
                        (): string;
                    };
                    var TextDecoration : {
                        (): string;
                    };
                    var TextRendering : {
                        (): string;
                    };
                    var To : {
                        (): string;
                    };
                    var Transform : {
                        (): string;
                    };
                    var Type : {
                        (): string;
                    };
                    var Values : {
                        (): string;
                    };
                    var ViewBox : {
                        (): string;
                    };
                    var Visibility : {
                        (): string;
                    };
                    var Width : {
                        (): string;
                    };
                    var WordSpacing : {
                        (): string;
                    };
                    var WritingMode : {
                        (): string;
                    };
                    var X : {
                        (): string;
                    };
                    var X1 : {
                        (): string;
                    };
                    var X2 : {
                        (): string;
                    };
                    var XChannelSelector : {
                        (): string;
                    };
                    var Y : {
                        (): string;
                    };
                    var Y1 : {
                        (): string;
                    };
                    var Y2 : {
                        (): string;
                    };
                    var YChannelSelector : {
                        (): string;
                    };
                    var Z : {
                        (): string;
                    };
                }
                module SvgElements {
                    var A : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var AltGlyph : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var AltGlyphDef : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var AltGlyphItem : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Animate : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var AnimateColor : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var AnimateMotion : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var AnimateTransform : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Circle : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var ClipPath : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var ColorProfile : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Cursor : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Defs : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Desc : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Ellipse : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var FeBlend : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var FeColorMatrix : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var FeComponentTransfer : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var FeComposite : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var FeConvolveMatrix : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var FeDiffuseLighting : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var FeDisplacementMap : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var FeDistantLight : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var FeFlood : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var FeFuncA : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var FeFuncB : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var FeFuncG : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var FeFuncR : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var FeGaussianBlur : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var FeImage : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var FeMerge : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var FeMergeNode : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var FeMorphology : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var FeOffset : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var FePointLight : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var FeSpecularLighting : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var FeSpotLight : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var FeTile : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var FeTurbulence : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Filter : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Font : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var FontFace : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var FontFaceFormat : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var FontFaceName : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var FontFaceSrc : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var FontFaceUri : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var ForeignObject : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var G : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Glyph : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var GlyphRef : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var HKern : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Image : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Line : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var LinearGradient : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Marker : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Mask : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Metadata : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var MissingGlyph : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var MPath : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Path : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Pattern : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Polygon : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Polyline : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var RadialGradient : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Rect : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Script : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Set : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Stop : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Style : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Svg : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Switch : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Symbol : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Text : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var TextPath : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Title : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var TRef : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var TSpan : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Use : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var View : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var VKern : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                }
                module Attributes {
                    var Accept : {
                        (): string;
                    };
                    var AcceptCharset : {
                        (): string;
                    };
                    var Accesskey : {
                        (): string;
                    };
                    var Action : {
                        (): string;
                    };
                    var Align : {
                        (): string;
                    };
                    var Alt : {
                        (): string;
                    };
                    var Async : {
                        (): string;
                    };
                    var AutoComplete : {
                        (): string;
                    };
                    var AutoFocus : {
                        (): string;
                    };
                    var AutoPlay : {
                        (): string;
                    };
                    var AutoSave : {
                        (): string;
                    };
                    var BgColor : {
                        (): string;
                    };
                    var Border : {
                        (): string;
                    };
                    var Buffered : {
                        (): string;
                    };
                    var Challenge : {
                        (): string;
                    };
                    var Charset : {
                        (): string;
                    };
                    var Checked : {
                        (): string;
                    };
                    var Cite : {
                        (): string;
                    };
                    var Class : {
                        (): string;
                    };
                    var Code : {
                        (): string;
                    };
                    var Codebase : {
                        (): string;
                    };
                    var Color : {
                        (): string;
                    };
                    var Cols : {
                        (): string;
                    };
                    var ColSpan : {
                        (): string;
                    };
                    var Content : {
                        (): string;
                    };
                    var ContentEditable : {
                        (): string;
                    };
                    var ContextMenu : {
                        (): string;
                    };
                    var Controls : {
                        (): string;
                    };
                    var Coords : {
                        (): string;
                    };
                    var Datetime : {
                        (): string;
                    };
                    var Default : {
                        (): string;
                    };
                    var Defer : {
                        (): string;
                    };
                    var Dir : {
                        (): string;
                    };
                    var DirName : {
                        (): string;
                    };
                    var Disabled : {
                        (): string;
                    };
                    var Download : {
                        (): string;
                    };
                    var Draggable : {
                        (): string;
                    };
                    var Dropzone : {
                        (): string;
                    };
                    var EncType : {
                        (): string;
                    };
                    var For : {
                        (): string;
                    };
                    var Form : {
                        (): string;
                    };
                    var FormAction : {
                        (): string;
                    };
                    var Headers : {
                        (): string;
                    };
                    var Height : {
                        (): string;
                    };
                    var Hidden : {
                        (): string;
                    };
                    var High : {
                        (): string;
                    };
                    var Href : {
                        (): string;
                    };
                    var HrefLang : {
                        (): string;
                    };
                    var HttpEquiv : {
                        (): string;
                    };
                    var Icon : {
                        (): string;
                    };
                    var ID : {
                        (): string;
                    };
                    var IsMap : {
                        (): string;
                    };
                    var ItemProp : {
                        (): string;
                    };
                    var KeyType : {
                        (): string;
                    };
                    var Kind : {
                        (): string;
                    };
                    var Label : {
                        (): string;
                    };
                    var Lang : {
                        (): string;
                    };
                    var Language : {
                        (): string;
                    };
                    var List : {
                        (): string;
                    };
                    var Loop : {
                        (): string;
                    };
                    var Low : {
                        (): string;
                    };
                    var Manifest : {
                        (): string;
                    };
                    var Max : {
                        (): string;
                    };
                    var MaxLength : {
                        (): string;
                    };
                    var Media : {
                        (): string;
                    };
                    var Method : {
                        (): string;
                    };
                    var Min : {
                        (): string;
                    };
                    var Multiple : {
                        (): string;
                    };
                    var Name : {
                        (): string;
                    };
                    var NoValidate : {
                        (): string;
                    };
                    var Open : {
                        (): string;
                    };
                    var Optimum : {
                        (): string;
                    };
                    var Pattern : {
                        (): string;
                    };
                    var Ping : {
                        (): string;
                    };
                    var Placeholder : {
                        (): string;
                    };
                    var Poster : {
                        (): string;
                    };
                    var Preload : {
                        (): string;
                    };
                    var PubDate : {
                        (): string;
                    };
                    var RadioGroup : {
                        (): string;
                    };
                    var Readonly : {
                        (): string;
                    };
                    var Rel : {
                        (): string;
                    };
                    var Required : {
                        (): string;
                    };
                    var Reversed : {
                        (): string;
                    };
                    var Rows : {
                        (): string;
                    };
                    var RowSpan : {
                        (): string;
                    };
                    var Sandbox : {
                        (): string;
                    };
                    var Spellcheck : {
                        (): string;
                    };
                    var Scope : {
                        (): string;
                    };
                    var Scoped : {
                        (): string;
                    };
                    var Seamless : {
                        (): string;
                    };
                    var Selected : {
                        (): string;
                    };
                    var Shape : {
                        (): string;
                    };
                    var Size : {
                        (): string;
                    };
                    var Sizes : {
                        (): string;
                    };
                    var Span : {
                        (): string;
                    };
                    var Src : {
                        (): string;
                    };
                    var Srcdoc : {
                        (): string;
                    };
                    var SrcLang : {
                        (): string;
                    };
                    var Start : {
                        (): string;
                    };
                    var Step : {
                        (): string;
                    };
                    var Style : {
                        (): string;
                    };
                    var Summary : {
                        (): string;
                    };
                    var TabIndex : {
                        (): string;
                    };
                    var Target : {
                        (): string;
                    };
                    var Title : {
                        (): string;
                    };
                    var Type : {
                        (): string;
                    };
                    var Usemap : {
                        (): string;
                    };
                    var Value : {
                        (): string;
                    };
                    var Width : {
                        (): string;
                    };
                    var Wrap : {
                        (): string;
                    };
                }
                module Elements {
                    var A : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Abbr : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Address : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Area : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Article : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Aside : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Audio : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var B : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Base : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var BDI : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var BDO : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var BlockQuote : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Body : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Br : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Button : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Canvas : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Caption : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Cite : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Code : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Col : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var ColGroup : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Data : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var DataList : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var DD : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Del : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Details : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var DFN : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Div : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var DL : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var DT : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Em : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Embed : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var FieldSet : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var FigCaption : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Figure : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Footer : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Form : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var H1 : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var H2 : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var H3 : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var H4 : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var H5 : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var H6 : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Head : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Header : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var HR : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Html : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var I : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var IFrame : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Img : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Input : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Ins : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Kbd : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Keygen : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Label : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Legend : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var LI : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Link : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Main : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Map : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Mark : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Menu : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var MenuItem : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Meta : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Meter : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Nav : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var NoScript : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Object : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var OL : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var OptGroup : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Option : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Output : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var P : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Param : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Picture : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Pre : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Progress : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Q : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var RP : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var RT : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Ruby : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var S : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Samp : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Script : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Section : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Select : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Small : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Source : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Span : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Strong : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Style : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Sub : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Summary : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Sup : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Table : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var TBody : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var TD : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var TextArea : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var TFoot : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var TH : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var THead : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Time : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Title : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var TR : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Track : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var U : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var UL : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Var : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var Video : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                    var WBR : {
                        (ats: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                    };
                }
                var A : {
                    (atr: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var A0 : {
                    (ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var Del : {
                    (atr: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var Del0 : {
                    (ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var Div : {
                    (atr: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var Div0 : {
                    (ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var Form : {
                    (atr: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var Form0 : {
                    (ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var H1 : {
                    (atr: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var H10 : {
                    (ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var H2 : {
                    (atr: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var H20 : {
                    (ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var H3 : {
                    (atr: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var H30 : {
                    (ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var H4 : {
                    (atr: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var H40 : {
                    (ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var H5 : {
                    (atr: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var H50 : {
                    (ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var H6 : {
                    (atr: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var H60 : {
                    (ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var LI : {
                    (atr: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var LI0 : {
                    (ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var Label : {
                    (atr: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var Label0 : {
                    (ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var Nav : {
                    (atr: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var Nav0 : {
                    (ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var P : {
                    (atr: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var P0 : {
                    (ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var Span : {
                    (atr: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var Span0 : {
                    (ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var Table : {
                    (atr: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var Table0 : {
                    (ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var TBody : {
                    (atr: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var TBody0 : {
                    (ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var THead : {
                    (atr: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var THead0 : {
                    (ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var TR : {
                    (atr: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var TR0 : {
                    (ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var TD : {
                    (atr: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var TD0 : {
                    (ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var UL : {
                    (atr: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var UL0 : {
                    (ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var OL : {
                    (atr: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Attr>, ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
                var OL0 : {
                    (ch: __ABBREV.__WebSharper.seq<__ABBREV.__Next.Doc>): __ABBREV.__Next.Doc;
                };
            }
            interface Var<_T1> {
                get_View(): __ABBREV.__Next.View<_T1>;
                get_Value(): _T1;
                set_Value(value: _T1): void;
                Const: boolean;
                Current: _T1;
                Snap: any;
                Id: number;
            }
            interface View<_T1> {
            }
            interface Var1 {
            }
            interface View1 {
            }
            interface ViewBuilder {
                Bind<_M1, _M2>(x: __ABBREV.__Next.View<_M1>, f: {
                    (x: _M1): __ABBREV.__Next.View<_M2>;
                }): __ABBREV.__Next.View<_M2>;
                Return<_M1>(x: _M1): __ABBREV.__Next.View<_M1>;
            }
            interface Key {
            }
            interface Model<_T1, _T2> {
                get_View(): __ABBREV.__Next.View<_T1>;
            }
            interface Model1 {
            }
            interface ListModel<_T1, _T2> {
                Add(item: _T2): void;
                Remove(item: _T2): void;
                RemoveByKey(key: _T1): void;
                Iter(fn: {
                    (x: _T2): void;
                }): void;
                Set(lst: __ABBREV.__WebSharper.seq<_T2>): void;
                ContainsKey(key: _T1): boolean;
                FindByKey(key: _T1): _T2;
                UpdateBy(fn: {
                    (x: _T2): __ABBREV.__WebSharper.OptionProxy<_T2>;
                }, key: _T1): void;
                Clear(): void;
                Key: {
                    (x: _T2): _T1;
                };
                Var: __ABBREV.__Next.Var<_T2[]>;
                View: __ABBREV.__Next.View<__ABBREV.__WebSharper.seq<_T2>>;
            }
            interface ListModel1 {
            }
            interface Interpolation<_T1> {
                Interpolate(x0: number, x1: _T1, x2: _T1): _T1;
            }
            interface Interpolation1 {
            }
            interface Easing {
                TransformTime: {
                    (x: number): number;
                };
            }
            interface Anim<_T1> {
                Compute: {
                    (x: number): _T1;
                };
                Duration: number;
            }
            interface An {
            }
            interface Trans<_T1> {
                TChange: {
                    (x: _T1): {
                        (x: _T1): any;
                    };
                };
                TEnter: {
                    (x: _T1): any;
                };
                TExit: {
                    (x: _T1): any;
                };
                TFlags: any;
            }
            interface Trans1 {
            }
            interface Attr {
                Flags: any;
                Tree: __ABBREV.__Next.AttrTree;
            }
            interface Doc {
                DocNode: __ABBREV.__Next.DocNode;
                Updates: __ABBREV.__Next.View<void>;
            }
            interface Flow<_T1> {
                Render: {
                    (x: __ABBREV.__Next.Var<__ABBREV.__Next.Doc>): {
                        (x: {
                            (x: _T1): void;
                        }): void;
                    };
                };
            }
            interface Flow1 {
            }
            interface FlowBuilder {
                Bind<_M1, _M2>(comp: any, func: {
                    (x: _M1): any;
                }): any;
                Return<_M1>(value: _M1): any;
                ReturnFrom<_M1>(inner: any): any;
            }
            interface RouteMap<_T1> {
                Des: {
                    (x: __ABBREV.__List.T<string>): _T1;
                };
                Ser: {
                    (x: _T1): __ABBREV.__List.T<string>;
                };
            }
            interface RouteId {
            }
            interface Router<_T1> {
            }
            interface RouteMap1 {
            }
            interface Router1 {
            }
        }
    }
}
declare module __ABBREV {
    
    export import __Next = WebSharper.UI.Next;
    export import __WebSharper = WebSharper;
    export import __Dom = WebSharper.JavaScript.Dom;
    export import __Client = WebSharper.Html.Client;
    export import __List = WebSharper.List;
}
