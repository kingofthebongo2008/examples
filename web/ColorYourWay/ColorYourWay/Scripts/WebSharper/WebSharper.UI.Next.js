(function()
{
 var Global=this,Runtime=this.IntelliFactory.Runtime,Arrays,Concurrency,Array,Seq,UI,Next,Abbrev,Fresh,Collections,HashSet,HashSet1,HashSet2,JQueue,Slot1,Unchecked,An,AppendList1,Anims,window,Trans,Option,View,Lazy,Array1,Attrs,DomUtility,Attr,AnimatedAttrNode,DynamicAttrNode,View1,Docs,UINextPagelet,Doc,List,Var1,Var,Mailbox,Operators,NodeSet,DocElemNode,DomNodes,jQuery,document,Easing,Easings,FlowBuilder,Flow1,Html,Elements,T,Input,DoubleInterpolation,Key,ListModels,ListModel,Model,Strings,encodeURIComponent,Route,decodeURIComponent,Routing,Router1,Router,Trie1,Dictionary,Snap1,Async,Enumerator,ResizeArray,ResizeArrayProxy,MapModule,FSharpMap,Html1,Client,Pagelet,Default,ViewBuilder,Attributes,SvgAttributes;
 Runtime.Define(Global,{
  WebSharper:{
   UI:{
    Next:{
     Abbrev:{
      Array:{
       MapReduce:function(f,z,re,a)
       {
        var loop;
        loop=function(off,len)
        {
         var l2,a1,b,l21,a2,b1;
         if(len<=0)
          {
           return z;
          }
         else
          {
           if(len===1)
            {
             if(off>=0?off<Arrays.length(a):false)
              {
               return f(Arrays.get(a,off));
              }
             else
              {
               l2=len/2>>0;
               a1=loop(off,l2);
               b=loop(off+l2,len-l2);
               return(re(a1))(b);
              }
            }
           else
            {
             l21=len/2>>0;
             a2=loop(off,l21);
             b1=loop(off+l21,len-l21);
             return(re(a2))(b1);
            }
          }
        };
        return loop(0,Arrays.length(a));
       }
      },
      Async:{
       Schedule:function(f)
       {
        var f1,arg00,t;
        f1=function()
        {
         var x;
         x=f(null);
         return Concurrency.Return(x);
        };
        arg00=Concurrency.Delay(f1);
        t={
         $:0
        };
        return Concurrency.Start(arg00,t);
       },
       StartTo:function(comp,k)
       {
        var c2,c3,t;
        c2=function()
        {
        };
        c3=function()
        {
        };
        t={
         $:0
        };
        return Concurrency.StartWithContinuations(comp,k,c2,c3,t);
       }
      },
      Dict:{
       ToKeyArray:function(d)
       {
        var arr;
        arr=Array(d.count);
        Seq.iteri(function(i)
        {
         return function(kv)
         {
          return Arrays.set(arr,i,kv.K);
         };
        },d);
        return arr;
       },
       ToValueArray:function(d)
       {
        var arr;
        arr=Array(d.count);
        Seq.iteri(function(i)
        {
         return function(kv)
         {
          return Arrays.set(arr,i,kv.V);
         };
        },d);
        return arr;
       }
      },
      Fresh:{
       Id:function()
       {
        var _;
        _=Fresh.counter()+1;
        Fresh.counter=function()
        {
         return _;
        };
        return"uid"+Global.String(Fresh.counter());
       },
       Int:function()
       {
        var _;
        _=Fresh.counter()+1;
        Fresh.counter=function()
        {
         return _;
        };
        return Fresh.counter();
       },
       counter:Runtime.Field(function()
       {
        return 0;
       })
      },
      HashSet:{
       Except:function(excluded,included)
       {
        var set;
        set=HashSet1.New3(HashSet2.ToArray(included));
        set.ExceptWith(HashSet2.ToArray(excluded));
        return set;
       },
       Filter:function(ok,set)
       {
        return HashSet1.New3(Arrays.filter(ok,HashSet2.ToArray(set)));
       },
       Intersect:function(a,b)
       {
        var set;
        set=HashSet1.New3(HashSet2.ToArray(a));
        set.IntersectWith(HashSet2.ToArray(b));
        return set;
       },
       ToArray:function(set)
       {
        var arr;
        arr=Array(set.get_Count());
        set.CopyTo(arr);
        return arr;
       }
      },
      JQueue:{
       Add:function($x,$q)
       {
        var $0=this,$this=this;
        return $q.push($x);
       },
       Count:function(q)
       {
        return q.length;
       },
       Dequeue:function($q)
       {
        var $0=this,$this=this;
        return $q.shift();
       },
       Iter:function(f,q)
       {
        return Arrays.iter(f,JQueue.ToArray(q));
       },
       ToArray:function(q)
       {
        return q.slice();
       }
      },
      Mailbox:{
       StartProcessor:function(proc)
       {
        var mail,isActive,f,work;
        mail=[];
        isActive={
         contents:false
        };
        f=function()
        {
         var g,f1,b,a,f3,b1;
         g=function()
         {
          return JQueue.Count(mail)>0;
         };
         f1=function()
         {
          var x,f2;
          x=proc(JQueue.Dequeue(mail));
          f2=function()
          {
           return Concurrency.Return(null);
          };
          return Concurrency.Bind(x,f2);
         };
         b=Concurrency.Delay(f1);
         a=Concurrency.While(g,b);
         f3=function()
         {
          var x;
          x=void(isActive.contents=false);
          return Concurrency.Return(x);
         };
         b1=Concurrency.Delay(f3);
         return Concurrency.Combine(a,b1);
        };
        work=Concurrency.Delay(f);
        return function(msg)
        {
         var t;
         JQueue.Add(msg,mail);
         if(!isActive.contents)
          {
           isActive.contents=true;
           t={
            $:0
           };
           return Concurrency.Start(work,t);
          }
         else
          {
           return null;
          }
        };
       }
      },
      Slot:Runtime.Class({},{
       Create:function(key,value)
       {
        return Slot1.New(key,value);
       }
      }),
      Slot1:Runtime.Class({
       Equals:function(o)
       {
        return Unchecked.Equals(this.key.call(null,this.value),this.key.call(null,o.get_Value()));
       },
       GetHashCode:function()
       {
        var x;
        x=this.key.call(null,this.value);
        return Unchecked.Hash(x);
       },
       get_Value:function()
       {
        return this.value;
       }
      },{
       New:function(key,value)
       {
        var r;
        r=Runtime.New(this,{});
        r.key=key;
        r.value=value;
        return r;
       }
      }),
      U:function()
      {
       return;
      }
     },
     An:Runtime.Class({},{
      Append:function(_arg2,_arg1)
      {
       return Runtime.New(An,{
        $:0,
        $0:AppendList1.Append(_arg2.$0,_arg1.$0)
       });
      },
      Concat:function(xs)
      {
       return Runtime.New(An,{
        $:0,
        $0:AppendList1.Concat(Seq.map(function(_arg00_)
        {
         return Anims.List(_arg00_);
        },xs))
       });
      },
      Const:function(v)
      {
       return Anims.Const(v);
      },
      Delayed:function(inter,easing,dur,delay,x,y)
      {
       return{
        Compute:function(t)
        {
         return t<=delay?x:inter.Interpolate(easing.TransformTime.call(null,(t-delay)/dur),x,y);
        },
        Duration:dur+delay
       };
      },
      Map:function(f,anim)
      {
       var f1;
       f1=anim.Compute;
       return Anims.Def(anim.Duration,function(x)
       {
        return f(f1(x));
       });
      },
      Pack:function(anim)
      {
       return Runtime.New(An,{
        $:0,
        $0:AppendList1.Single({
         $:1,
         $0:anim
        })
       });
      },
      Play:function(anim)
      {
       var f;
       f=function()
       {
        var x,f1;
        x=An.Run(function()
        {
        },Anims.Actions(anim));
        f1=function()
        {
         var x1;
         x1=Anims.Finalize(anim);
         return Concurrency.Return(x1);
        };
        return Concurrency.Bind(x,f1);
       };
       return Concurrency.Delay(f);
      },
      Run:function(k,anim)
      {
       var dur,arg00;
       dur=anim.Duration;
       arg00=function(tupledArg)
       {
        var ok,start,loop;
        ok=tupledArg[0];
        start=function()
        {
         window.requestAnimationFrame(function(t)
         {
          return loop(t,t);
         });
        };
        loop=function(start1,now)
        {
         var t;
         t=now-start1;
         k(anim.Compute.call(null,t));
         return t<=dur?void window.requestAnimationFrame(function(t1)
         {
          return loop(start1,t1);
         }):ok(null);
        };
        return start(null);
       };
       return Concurrency.FromContinuations(arg00);
      },
      Simple:function(inter,easing,dur,x,y)
      {
       return{
        Compute:function(t)
        {
         return inter.Interpolate(easing.TransformTime.call(null,t/dur),x,y);
        },
        Duration:dur
       };
      },
      WhenDone:function(f,main)
      {
       return An.Append(Runtime.New(An,{
        $:0,
        $0:AppendList1.Single({
         $:0,
         $0:f
        })
       }),main);
      },
      get_Empty:function()
      {
       return Runtime.New(An,{
        $:0,
        $0:AppendList1.Empty()
       });
      }
     }),
     AnimatedAttrNode:Runtime.Class({
      GetChangeAnim:function(parent)
      {
       var matchValue,a=this;
       matchValue=[this.visible,this.logical];
       return An.WhenDone(function()
       {
        return a.sync(parent);
       },matchValue[0].$==1?matchValue[1].$==1?a.dirty?An.Pack(An.Map(function(v)
       {
        return a.pushVisible(parent,v);
       },Trans.AnimateChange(a.tr,matchValue[0].$0,matchValue[1].$0))):An.get_Empty():An.get_Empty():An.get_Empty());
      },
      GetEnterAnim:function(parent)
      {
       var matchValue,a=this;
       matchValue=[this.visible,this.logical];
       return An.WhenDone(function()
       {
        return a.sync(parent);
       },matchValue[0].$==1?matchValue[1].$==1?a.dirty?An.Pack(An.Map(function(v)
       {
        return a.pushVisible(parent,v);
       },Trans.AnimateChange(a.tr,matchValue[0].$0,matchValue[1].$0))):matchValue[0].$==0?matchValue[1].$==1?An.Pack(An.Map(function(v)
       {
        return a.pushVisible(parent,v);
       },Trans.AnimateEnter(a.tr,matchValue[1].$0))):An.get_Empty():An.get_Empty():matchValue[0].$==0?matchValue[1].$==1?An.Pack(An.Map(function(v)
       {
        return a.pushVisible(parent,v);
       },Trans.AnimateEnter(a.tr,matchValue[1].$0))):An.get_Empty():An.get_Empty():matchValue[0].$==0?matchValue[1].$==1?An.Pack(An.Map(function(v)
       {
        return a.pushVisible(parent,v);
       },Trans.AnimateEnter(a.tr,matchValue[1].$0))):An.get_Empty():An.get_Empty());
      },
      GetExitAnim:function(parent)
      {
       var matchValue,a=this;
       matchValue=this.visible;
       return An.WhenDone(function()
       {
        a.dirty=true;
        a.visible={
         $:0
        };
        return;
       },matchValue.$==1?An.Pack(An.Map(function(v)
       {
        return a.pushVisible(parent,v);
       },Trans.AnimateExit(a.tr,matchValue.$0))):An.get_Empty());
      },
      Sync:function()
      {
       return null;
      },
      get_Changed:function()
      {
       return this.updates;
      },
      pushVisible:function(el,v)
      {
       this.visible={
        $:1,
        $0:v
       };
       this.dirty=true;
       return(this.push.call(null,el))(v);
      },
      sync:function(p)
      {
       if(this.dirty)
        {
         Option.iter(this.push.call(null,p),this.logical);
         this.visible=this.logical;
         this.dirty=false;
         return;
        }
       else
        {
         return null;
        }
      }
     },{
      New:function(tr,view,push)
      {
       var r;
       r=Runtime.New(this,{});
       r.tr=tr;
       r.push=push;
       r.logical={
        $:0
       };
       r.visible={
        $:0
       };
       r.dirty=true;
       r.updates=View.Map(function(x)
       {
        r.logical={
         $:1,
         $0:x
        };
        r.dirty=true;
        return;
       },view);
       return r;
      }
     }),
     Anims:{
      Actions:function(_arg1)
      {
       return Anims.ConcatActions(Arrays.choose(function(_arg2)
       {
        return _arg2.$==1?{
         $:1,
         $0:_arg2.$0
        }:{
         $:0
        };
       },AppendList1.ToArray(_arg1.$0)));
      },
      ConcatActions:function(xs)
      {
       var xs1,matchValue,dur,xs2;
       xs1=Seq.toArray(xs);
       matchValue=Arrays.length(xs1);
       if(matchValue===0)
        {
         return Anims.Const(null);
        }
       else
        {
         if(matchValue===1)
          {
           return Arrays.get(xs1,0);
          }
         else
          {
           dur=Seq.max(Seq.map(function(anim)
           {
            return anim.Duration;
           },xs1));
           xs2=Arrays.map(function(anim)
           {
            return Anims.Prolong(dur,anim);
           },xs1);
           return Anims.Def(dur,function(t)
           {
            return Arrays.iter(function(anim)
            {
             return anim.Compute.call(null,t);
            },xs2);
           });
          }
        }
      },
      Const:function(v)
      {
       return Anims.Def(0,function()
       {
        return v;
       });
      },
      Def:function(d,f)
      {
       return{
        Compute:f,
        Duration:d
       };
      },
      Finalize:function(_arg1)
      {
       return Arrays.iter(function(_arg2)
       {
        return _arg2.$==0?_arg2.$0.call(null,null):null;
       },AppendList1.ToArray(_arg1.$0));
      },
      List:function(_arg1)
      {
       return _arg1.$0;
      },
      Prolong:function(nextDuration,anim)
      {
       var comp,dur,last;
       comp=anim.Compute;
       dur=anim.Duration;
       last=Lazy.Create(function()
       {
        return anim.Compute.call(null,anim.Duration);
       });
       return{
        Compute:function(t)
        {
         return t>=dur?last.eval():comp(t);
        },
        Duration:nextDuration
       };
      }
     },
     AppendList1:{
      Append:function(x,y)
      {
       var matchValue;
       matchValue=[x,y];
       return matchValue[0].$==0?matchValue[1]:matchValue[1].$==0?matchValue[0]:{
        $:2,
        $0:x,
        $1:y
       };
      },
      Concat:function(xs)
      {
       var a;
       a=Seq.toArray(xs);
       return Array1.MapReduce(function(x)
       {
        return x;
       },AppendList1.Empty(),function(_arg00_)
       {
        return function(_arg10_)
        {
         return AppendList1.Append(_arg00_,_arg10_);
        };
       },a);
      },
      Empty:function()
      {
       return{
        $:0
       };
      },
      FromArray:function(xs)
      {
       var matchValue;
       matchValue=xs.length;
       return matchValue===0?{
        $:0
       }:matchValue===1?{
        $:1,
        $0:Arrays.get(xs,0)
       }:{
        $:3,
        $0:xs.slice()
       };
      },
      Single:function(x)
      {
       return{
        $:1,
        $0:x
       };
      },
      ToArray:function(xs)
      {
       var out,loop;
       out=[];
       loop=function(xs1)
       {
        var y;
        if(xs1.$==1)
         {
          return JQueue.Add(xs1.$0,out);
         }
        else
         {
          if(xs1.$==2)
           {
            y=xs1.$1;
            loop(xs1.$0);
            return loop(y);
           }
          else
           {
            return xs1.$==3?Arrays.iter(function(v)
            {
             return JQueue.Add(v,out);
            },xs1.$0):null;
           }
         }
       };
       loop(xs);
       return JQueue.ToArray(out);
      }
     },
     Attr:Runtime.Class({},{
      Animated:function(name,tr,view,attr)
      {
       return Attrs.Animated(tr,view,function(el)
       {
        return function(v)
        {
         return DomUtility.SetAttr(el,name,attr(v));
        };
       });
      },
      AnimatedStyle:function(name,tr,view,attr)
      {
       return Attrs.Animated(tr,view,function(el)
       {
        return function(v)
        {
         return DomUtility.SetStyle(el,name,attr(v));
        };
       });
      },
      Append:function(a,b)
      {
       return Attrs.Mk(a.Flags|b.Flags,Attrs.AppendTree(a.Tree,b.Tree));
      },
      Class:function(name)
      {
       return Attrs.Static(function(el)
       {
        return DomUtility.AddClass(el,name);
       });
      },
      Concat:function(xs)
      {
       var f,re,a;
       f=function(x)
       {
        return x;
       };
       re=function(arg00)
       {
        return function(arg10)
        {
         return Attr.Append(arg00,arg10);
        };
       };
       a=Seq.toArray(xs);
       return Array1.MapReduce(f,Attrs.EmptyAttr(),re,a);
      },
      Create:function(name,value)
      {
       return Attrs.Static(function(el)
       {
        return DomUtility.SetAttr(el,name,value);
       });
      },
      Dynamic:function(name,view)
      {
       return Attrs.Dynamic(view,function(el)
       {
        return function(v)
        {
         return DomUtility.SetAttr(el,name,v);
        };
       });
      },
      DynamicClass:function(name,view,ok)
      {
       return Attrs.Dynamic(view,function(el)
       {
        return function(v)
        {
         return ok(v)?DomUtility.AddClass(el,name):DomUtility.RemoveClass(el,name);
        };
       });
      },
      DynamicCustom:function(set,view)
      {
       return Attrs.Dynamic(view,set);
      },
      DynamicPred:function(name,predView,valView)
      {
       var viewFn;
       viewFn=function(el)
       {
        return function(tupledArg)
        {
         var v;
         v=tupledArg[1];
         return tupledArg[0]?DomUtility.SetAttr(el,name,v):DomUtility.RemoveAttr(el,name);
        };
       };
       return Attrs.Dynamic(View.Map2(function(pred)
       {
        return function(value)
        {
         return[pred,value];
        };
       },predView,valView),viewFn);
      },
      DynamicStyle:function(name,view)
      {
       return Attrs.Dynamic(view,function(el)
       {
        return function(v)
        {
         return DomUtility.SetStyle(el,name,v);
        };
       });
      },
      Handler:function(name,callback)
      {
       return Attrs.Static(function(el)
       {
        return el.addEventListener(name,callback,false);
       });
      },
      Style:function(name,value)
      {
       return Attrs.Static(function(el)
       {
        return DomUtility.SetStyle(el,name,value);
       });
      },
      get_Empty:function()
      {
       return Attrs.EmptyAttr();
      }
     }),
     Attrs:{
      Animated:function(tr,view,set)
      {
       var node,flags;
       node=AnimatedAttrNode.New(tr,view,set);
       flags=4;
       if(Trans.CanAnimateEnter(tr))
        {
         flags=flags|1;
        }
       if(Trans.CanAnimateExit(tr))
        {
         flags=flags|2;
        }
       return Attrs.Mk(flags,{
        $:1,
        $0:node
       });
      },
      AppendTree:function(a,b)
      {
       var matchValue;
       matchValue=[a,b];
       return matchValue[0].$==0?matchValue[1]:matchValue[1].$==0?matchValue[0]:{
        $:2,
        $0:a,
        $1:b
       };
      },
      Dynamic:function(view,set)
      {
       return Attrs.Mk(0,{
        $:1,
        $0:DynamicAttrNode.New(view,set)
       });
      },
      EmptyAttr:Runtime.Field(function()
      {
       return Attrs.Mk(0,{
        $:0
       });
      }),
      GetAnim:function(dyn,f)
      {
       return An.Concat(Arrays.map(function(n)
       {
        return(f(n))(dyn.DynElem);
       },dyn.DynNodes));
      },
      GetChangeAnim:function(dyn)
      {
       return Attrs.GetAnim(dyn,function(n)
       {
        return function(arg00)
        {
         return n.GetChangeAnim(arg00);
        };
       });
      },
      GetEnterAnim:function(dyn)
      {
       return Attrs.GetAnim(dyn,function(n)
       {
        return function(arg00)
        {
         return n.GetEnterAnim(arg00);
        };
       });
      },
      GetExitAnim:function(dyn)
      {
       return Attrs.GetAnim(dyn,function(n)
       {
        return function(arg00)
        {
         return n.GetExitAnim(arg00);
        };
       });
      },
      HasChangeAnim:function(attr)
      {
       return(attr.DynFlags&4)!==0;
      },
      HasEnterAnim:function(attr)
      {
       return(attr.DynFlags&1)!==0;
      },
      HasExitAnim:function(attr)
      {
       return(attr.DynFlags&2)!==0;
      },
      Insert:function(elem,tree)
      {
       var nodes,loop;
       nodes=[];
       loop=function(node)
       {
        var b;
        if(node.$==1)
         {
          return JQueue.Add(node.$0,nodes);
         }
        else
         {
          if(node.$==2)
           {
            b=node.$1;
            loop(node.$0);
            return loop(b);
           }
          else
           {
            return node.$==3?node.$0.call(null,elem):null;
           }
         }
       };
       loop(tree.Tree);
       return{
        DynElem:elem,
        DynFlags:tree.Flags,
        DynNodes:JQueue.ToArray(nodes)
       };
      },
      Mk:function(flags,tree)
      {
       return Runtime.New(Attr,{
        Flags:flags,
        Tree:tree
       });
      },
      Static:function(attr)
      {
       return Attrs.Mk(0,{
        $:3,
        $0:attr
       });
      },
      Sync:function(elem,dyn)
      {
       return Arrays.iter(function(d)
       {
        return d.Sync(elem);
       },dyn.DynNodes);
      },
      Updates:function(dyn)
      {
       var p,a;
       p=function(x)
       {
        return function(y)
        {
         return View.Map2(function()
         {
          return function()
          {
           return null;
          };
         },x,y);
        };
       };
       a=dyn.DynNodes;
       return Array1.MapReduce(function(x)
       {
        return x.get_Changed();
       },View1.Const(null),p,a);
      }
     },
     Doc:Runtime.Class({},{
      Append:function(a,b)
      {
       var x;
       x=View.Map2(function()
       {
        return function()
        {
         return null;
        };
       },a.Updates,b.Updates);
       return Docs.Mk({
        $:0,
        $0:a.DocNode,
        $1:b.DocNode
       },x);
      },
      AsPagelet:function(doc)
      {
       return UINextPagelet.New(doc);
      },
      Button:function(caption,attrs,action)
      {
       var attrs1;
       attrs1=Attr.Concat(attrs);
       return Doc.Elem(Doc.Clickable("button",action),attrs1,Doc.TextNode(caption));
      },
      CheckBox:function(attrs,item,chk)
      {
       var rvi,predicate,checkedView,arg20,checkedAttr,a,b,x1,attrs1,el;
       rvi=View.FromVar(chk);
       predicate=function(x)
       {
        return Unchecked.Equals(x,item);
       };
       checkedView=View.Map(function(list)
       {
        return Seq.exists(predicate,list);
       },rvi);
       arg20=View1.Const("checked");
       checkedAttr=Attr.DynamicPred("checked",checkedView,arg20);
       a=List.ofArray([Attr.Create("type","checkbox"),Attr.Create("name",Global.String(Var1.GetId(chk))),Attr.Create("value",Fresh.Id()),checkedAttr]);
       b=List.ofSeq(attrs);
       x1=List.append(a,b);
       attrs1=Attr.Concat(x1);
       el=DomUtility.CreateElement("input");
       el.addEventListener("click",function()
       {
        var chkd;
        chkd=el.checked;
        return Var1.Update(chk,function(obs)
        {
         var _,b1;
         if(chkd)
          {
           b1=List.ofArray([item]);
           _=List.append(obs,b1);
          }
         else
          {
           _=List.filter(function(x2)
           {
            return!Unchecked.Equals(x2,item);
           },obs);
          }
         return Seq.toList(Seq.distinct(_));
        });
       },false);
       return Doc.Elem(el,attrs1,Doc.get_Empty());
      },
      Clickable:function(elem,action)
      {
       var el;
       el=DomUtility.CreateElement(elem);
       el.addEventListener("click",function(ev)
       {
        ev.preventDefault();
        return action(null);
       },false);
       return el;
      },
      Concat:function(xs)
      {
       var a;
       a=Seq.toArray(xs);
       return Array1.MapReduce(function(x)
       {
        return x;
       },Doc.get_Empty(),function(arg00)
       {
        return function(arg10)
        {
         return Doc.Append(arg00,arg10);
        };
       },a);
      },
      Convert:function(render,view)
      {
       return Doc.Flatten(View.Convert(render,view));
      },
      ConvertBy:function(key,render,view)
      {
       return Doc.Flatten(View.ConvertBy(key,render,view));
      },
      ConvertSeq:function(render,view)
      {
       return Doc.Flatten(View1.ConvertSeq(render,view));
      },
      ConvertSeqBy:function(key,render,view)
      {
       return Doc.Flatten(View.ConvertSeqBy(key,render,view));
      },
      Elem:function(name,attr,children)
      {
       var node,arg20,updates;
       node=Docs.CreateElemNode(name,attr,children.DocNode);
       arg20=children.Updates;
       updates=View.Map2(function()
       {
        return function()
        {
         return null;
        };
       },Attrs.Updates(node.Attr),arg20);
       return Docs.Mk({
        $:1,
        $0:node
       },updates);
      },
      Element:function(name,attr,children)
      {
       var attr1,arg20;
       attr1=Attr.Concat(attr);
       arg20=Doc.Concat(children);
       return Doc.Elem(DomUtility.CreateElement(name),attr1,arg20);
      },
      EmbedView:function(view)
      {
       var node,x;
       node=Docs.CreateEmbedNode();
       x=View.Map(function()
       {
       },View1.Bind(function(doc)
       {
        Docs.UpdateEmbedNode(node,doc.DocNode);
        return doc.Updates;
       },view));
       return Docs.Mk({
        $:2,
        $0:node
       },x);
      },
      Flatten:function(view)
      {
       return Doc.EmbedView(View.Map(function(arg00)
       {
        return Doc.Concat(arg00);
       },view));
      },
      Input:function(attr,_var)
      {
       return Doc.InputInternal(attr,_var,{
        $:0
       });
      },
      InputArea:function(attr,_var)
      {
       return Doc.InputInternal(attr,_var,{
        $:2
       });
      },
      InputInternal:function(attr,_var,inputTy)
      {
       var patternInput,attrN,el,valAttr;
       patternInput=inputTy.$==1?[Attr.Append(Attr.Create("type","password"),Attr.Concat(attr)),"input"]:inputTy.$==2?[Attr.Concat(attr),"textarea"]:[Attr.Concat(attr),"input"];
       attrN=patternInput[0];
       el=DomUtility.CreateElement(patternInput[1]);
       valAttr=Attr.DynamicCustom(function(el1)
       {
        return function(v)
        {
         el1.value=v;
        };
       },View.FromVar(_var));
       el.addEventListener("input",function()
       {
        return Var.Set(_var,el.value);
       },false);
       return Doc.Elem(el,Attr.Append(attrN,valAttr),Doc.get_Empty());
      },
      Link:function(caption,attrs,action)
      {
       var arg10,attrs1;
       arg10=Attr.Concat(attrs);
       attrs1=Attr.Append(Attr.Create("href","#"),arg10);
       return Doc.Elem(Doc.Clickable("a",action),attrs1,Doc.TextNode(caption));
      },
      PasswordBox:function(attr,_var)
      {
       return Doc.InputInternal(attr,_var,{
        $:1
       });
      },
      Radio:function(attrs,value,_var)
      {
       var el,valAttr,op_EqualsEqualsGreater,a,b;
       el=DomUtility.CreateElement("input");
       el.addEventListener("click",function()
       {
        return Var.Set(_var,value);
       },false);
       valAttr=Attr.DynamicPred("checked",View.Map(function(x)
       {
        return Unchecked.Equals(x,value);
       },_var.get_View()),View1.Const("checked"));
       op_EqualsEqualsGreater=function(k,v)
       {
        return Attr.Create(k,v);
       };
       a=List.ofArray([op_EqualsEqualsGreater("type","radio"),op_EqualsEqualsGreater("name",Global.String(Var1.GetId(_var))),valAttr]);
       b=List.ofSeq(attrs);
       return Doc.Elem(el,Attr.Concat(List.append(a,b)),Doc.get_Empty());
      },
      Run:function(parent,doc)
      {
       var d,st,arg10;
       d=doc.DocNode;
       Docs.LinkElement(parent,d);
       st=Docs.CreateRunState(parent,d);
       arg10=doc.Updates;
       return View1.Sink(Mailbox.StartProcessor(function()
       {
        return Docs.PerformAnimatedUpdate(st,parent,d);
       }),arg10);
      },
      RunById:function(id,tr)
      {
       var matchValue;
       matchValue=DomUtility.Doc().getElementById(id);
       return Unchecked.Equals(matchValue,null)?Operators.FailWith("invalid id: "+id):Doc.Run(matchValue,tr);
      },
      Select:function(attrs,show,options,current)
      {
       var setSelectedItem,el1,x,selectedItemAttr,optionElements;
       setSelectedItem=function(el)
       {
        return function(item)
        {
         var p;
         p=function(y)
         {
          return Unchecked.Equals(item,y);
         };
         el.selectedIndex=Seq.findIndex(p,options);
         return;
        };
       };
       el1=DomUtility.CreateElement("select");
       x=View.FromVar(current);
       selectedItemAttr=Attr.DynamicCustom(setSelectedItem,x);
       el1.addEventListener("change",function()
       {
        return Var.Set(current,options.get_Item(el1.selectedIndex));
       },false);
       optionElements=Doc.Concat(List.mapi(function(i)
       {
        return function(o)
        {
         var t;
         t=Doc.TextNode(show(o));
         return Doc.Element("option",List.ofArray([Attr.Create("value",Global.String(i))]),List.ofArray([t]));
        };
       },options));
       return Doc.Elem(el1,Attr.Append(selectedItemAttr,Attr.Concat(attrs)),optionElements);
      },
      Static:function(el)
      {
       return Doc.Elem(el,Attr.get_Empty(),Doc.get_Empty());
      },
      SvgElement:function(name,attr,children)
      {
       var attr1,arg20;
       attr1=Attr.Concat(attr);
       arg20=Doc.Concat(children);
       return Doc.Elem(DomUtility.CreateSvgElement(name),attr1,arg20);
      },
      TextNode:function(v)
      {
       return Doc.TextView(View1.Const(v));
      },
      TextView:function(txt)
      {
       var node,x;
       node=Docs.CreateTextNode();
       x=View.Map(function(t)
       {
        return Docs.UpdateTextNode(node,t);
       },txt);
       return Docs.Mk({
        $:4,
        $0:node
       },x);
      },
      get_Empty:function()
      {
       return Docs.Mk({
        $:3
       },View1.Const(null));
      }
     }),
     DocElemNode:Runtime.Class({
      Equals:function(o)
      {
       return this.ElKey===o.ElKey;
      },
      GetHashCode:function()
      {
       return this.ElKey;
      }
     }),
     Docs:{
      ComputeChangeAnim:function(st,cur)
      {
       var arg00,relevant;
       arg00=function(n)
       {
        return Attrs.HasChangeAnim(n.Attr);
       };
       relevant=function(arg10)
       {
        return NodeSet.Filter(arg00,arg10);
       };
       return An.Concat(Arrays.map(function(n)
       {
        return Attrs.GetChangeAnim(n.Attr);
       },NodeSet.ToArray(NodeSet.Intersect(relevant(st.PreviousNodes),relevant(cur)))));
      },
      ComputeEnterAnim:function(st,cur)
      {
       return An.Concat(Arrays.map(function(n)
       {
        return Attrs.GetEnterAnim(n.Attr);
       },NodeSet.ToArray(NodeSet.Except(st.PreviousNodes,NodeSet.Filter(function(n)
       {
        return Attrs.HasEnterAnim(n.Attr);
       },cur)))));
      },
      ComputeExitAnim:function(st,cur)
      {
       return An.Concat(Arrays.map(function(n)
       {
        return Attrs.GetExitAnim(n.Attr);
       },NodeSet.ToArray(NodeSet.Except(cur,NodeSet.Filter(function(n)
       {
        return Attrs.HasExitAnim(n.Attr);
       },st.PreviousNodes)))));
      },
      CreateElemNode:function(el,attr,children)
      {
       Docs.LinkElement(el,children);
       return Runtime.New(DocElemNode,{
        Attr:Attrs.Insert(el,attr),
        Children:children,
        El:el,
        ElKey:Fresh.Int()
       });
      },
      CreateEmbedNode:function()
      {
       return{
        Current:{
         $:3
        },
        Dirty:false
       };
      },
      CreateRunState:function(parent,doc)
      {
       return{
        PreviousNodes:NodeSet.get_Empty(),
        Top:Docs.CreateElemNode(parent,Attr.get_Empty(),doc)
       };
      },
      CreateTextNode:function()
      {
       return{
        Text:DomUtility.CreateText(""),
        Dirty:false,
        Value:""
       };
      },
      DoSyncElement:function(el)
      {
       var parent,ins,parent1;
       parent=el.El;
       ins=function(doc,pos)
       {
        var d;
        if(doc.$==1)
         {
          return{
           $:1,
           $0:doc.$0.El
          };
         }
        else
         {
          if(doc.$==2)
           {
            d=doc.$0;
            if(d.Dirty)
             {
              d.Dirty=false;
              return Docs.InsertDoc(parent,d.Current,pos);
             }
            else
             {
              return ins(d.Current,pos);
             }
           }
          else
           {
            return doc.$==3?pos:doc.$==4?{
             $:1,
             $0:doc.$0.Text
            }:ins(doc.$0,ins(doc.$1,pos));
           }
         }
       };
       parent1=el.El;
       DomNodes.Iter(function(el1)
       {
        return DomUtility.RemoveNode(parent1,el1);
       },DomNodes.Except(DomNodes.DocChildren(el),DomNodes.Children(el.El)));
       ins(el.Children,{
        $:0
       });
       return;
      },
      DomNodes:Runtime.Class({},{
       Children:function(elem)
       {
        var objectArg;
        objectArg=elem.childNodes;
        return Runtime.New(DomNodes,{
         $:0,
         $0:Arrays.init(elem.childNodes.length,function(arg00)
         {
          return objectArg[arg00];
         })
        });
       },
       DocChildren:function(node)
       {
        var q,loop;
        q=[];
        loop=function(doc)
        {
         var b;
         if(doc.$==2)
          {
           return loop(doc.$0.Current);
          }
         else
          {
           if(doc.$==1)
            {
             return JQueue.Add(doc.$0.El,q);
            }
           else
            {
             if(doc.$==3)
              {
               return null;
              }
             else
              {
               if(doc.$==4)
                {
                 return JQueue.Add(doc.$0.Text,q);
                }
               else
                {
                 b=doc.$1;
                 loop(doc.$0);
                 return loop(b);
                }
              }
            }
          }
        };
        loop(node.Children);
        return Runtime.New(DomNodes,{
         $:0,
         $0:JQueue.ToArray(q)
        });
       },
       Except:function(_arg2,_arg1)
       {
        var excluded;
        excluded=_arg2.$0;
        return Runtime.New(DomNodes,{
         $:0,
         $0:Arrays.filter(function(n)
         {
          var predicate;
          predicate=function(k)
          {
           return!(n===k);
          };
          return Seq.forall(predicate,excluded);
         },_arg1.$0)
        });
       },
       FoldBack:function(f,_arg4,z)
       {
        return Arrays.foldBack(f,_arg4.$0,z);
       },
       Iter:function(f,_arg3)
       {
        return Arrays.iter(f,_arg3.$0);
       }
      }),
      InsertDoc:function(parent,doc,pos)
      {
       var d;
       if(doc.$==1)
        {
         return Docs.InsertNode(parent,doc.$0.El,pos);
        }
       else
        {
         if(doc.$==2)
          {
           d=doc.$0;
           d.Dirty=false;
           return Docs.InsertDoc(parent,d.Current,pos);
          }
         else
          {
           return doc.$==3?pos:doc.$==4?Docs.InsertNode(parent,doc.$0.Text,pos):Docs.InsertDoc(parent,doc.$0,Docs.InsertDoc(parent,doc.$1,pos));
          }
        }
      },
      InsertNode:function(parent,node,pos)
      {
       DomUtility.InsertAt(parent,pos,node);
       return{
        $:1,
        $0:node
       };
      },
      LinkElement:function(el,children)
      {
       Docs.InsertDoc(el,children,{
        $:0
       });
      },
      Mk:function(node,updates)
      {
       return Runtime.New(Doc,{
        DocNode:node,
        Updates:updates
       });
      },
      NodeSet:Runtime.Class({},{
       Except:function(_arg3,_arg2)
       {
        return Runtime.New(NodeSet,{
         $:0,
         $0:HashSet2.Except(_arg3.$0,_arg2.$0)
        });
       },
       Filter:function(f,_arg1)
       {
        return Runtime.New(NodeSet,{
         $:0,
         $0:HashSet2.Filter(f,_arg1.$0)
        });
       },
       FindAll:function(doc)
       {
        var q,loop;
        q=[];
        loop=function(node)
        {
         var b,el;
         if(node.$==0)
          {
           b=node.$1;
           loop(node.$0);
           return loop(b);
          }
         else
          {
           if(node.$==1)
            {
             el=node.$0;
             JQueue.Add(el,q);
             return loop(el.Children);
            }
           else
            {
             return node.$==2?loop(node.$0.Current):null;
            }
          }
        };
        loop(doc);
        return Runtime.New(NodeSet,{
         $:0,
         $0:HashSet1.New3(JQueue.ToArray(q))
        });
       },
       Intersect:function(_arg5,_arg4)
       {
        return Runtime.New(NodeSet,{
         $:0,
         $0:HashSet2.Intersect(_arg5.$0,_arg4.$0)
        });
       },
       IsEmpty:function(_arg6)
       {
        return _arg6.$0.get_Count()===0;
       },
       ToArray:function(_arg7)
       {
        return HashSet2.ToArray(_arg7.$0);
       },
       get_Empty:function()
       {
        return Runtime.New(NodeSet,{
         $:0,
         $0:HashSet1.New11()
        });
       }
      }),
      PerformAnimatedUpdate:function(st,parent,doc)
      {
       var f;
       f=function()
       {
        var cur,change,enter,x,f1;
        cur=NodeSet.FindAll(doc);
        change=Docs.ComputeChangeAnim(st,cur);
        enter=Docs.ComputeEnterAnim(st,cur);
        x=An.Play(An.Append(change,Docs.ComputeExitAnim(st,cur)));
        f1=function()
        {
         var x1,f2;
         Docs.SyncElemNode(st.Top);
         x1=An.Play(enter);
         f2=function()
         {
          var x2;
          x2=void(st.PreviousNodes=cur);
          return Concurrency.Return(x2);
         };
         return Concurrency.Bind(x1,f2);
        };
        return Concurrency.Bind(x,f1);
       };
       return Concurrency.Delay(f);
      },
      Sync:function(doc)
      {
       var sync;
       sync=function(doc1)
       {
        var el,d,b;
        if(doc1.$==1)
         {
          el=doc1.$0;
          Docs.SyncElement(el);
          return sync(el.Children);
         }
        else
         {
          if(doc1.$==2)
           {
            return sync(doc1.$0.Current);
           }
          else
           {
            if(doc1.$==3)
             {
              return null;
             }
            else
             {
              if(doc1.$==4)
               {
                d=doc1.$0;
                if(d.Dirty)
                 {
                  d.Text.nodeValue=d.Value;
                  d.Dirty=false;
                  return;
                 }
                else
                 {
                  return null;
                 }
               }
              else
               {
                b=doc1.$1;
                sync(doc1.$0);
                return sync(b);
               }
             }
           }
         }
       };
       return sync(doc);
      },
      SyncElemNode:function(el)
      {
       Docs.SyncElement(el);
       return Docs.Sync(el.Children);
      },
      SyncElement:function(el)
      {
       var dirty;
       Attrs.Sync(el.El,el.Attr);
       dirty=function(doc)
       {
        var b,d;
        if(doc.$==0)
         {
          b=doc.$1;
          return dirty(doc.$0)?true:dirty(b);
         }
        else
         {
          if(doc.$==2)
           {
            d=doc.$0;
            return d.Dirty?true:dirty(d.Current);
           }
          else
           {
            return false;
           }
         }
       };
       return dirty(el.Children)?Docs.DoSyncElement(el):null;
      },
      UpdateEmbedNode:function(node,upd)
      {
       node.Current=upd;
       node.Dirty=true;
       return;
      },
      UpdateTextNode:function(n,t)
      {
       n.Value=t;
       n.Dirty=true;
       return;
      }
     },
     DomUtility:{
      AddClass:function(element,cl)
      {
       jQuery(element).addClass(cl);
      },
      AppendTo:function(ctx,node)
      {
       ctx.appendChild(node);
      },
      Clear:function(ctx)
      {
       while(ctx.hasChildNodes())
        {
         ctx.removeChild(ctx.firstChild);
        }
       return;
      },
      ClearAttrs:function(ctx)
      {
       while(ctx.hasAttributes())
        {
         ctx.removeAttributeNode(ctx.attributes.item(0));
        }
       return;
      },
      CreateAttr:function(name,value)
      {
       var a;
       a=DomUtility.Doc().createAttribute(name);
       a.value=value;
       return a;
      },
      CreateElement:function(name)
      {
       return DomUtility.Doc().createElement(name);
      },
      CreateSvgElement:function(name)
      {
       return DomUtility.Doc().createElementNS("http://www.w3.org/2000/svg",name);
      },
      CreateText:function(s)
      {
       return DomUtility.Doc().createTextNode(s);
      },
      Doc:Runtime.Field(function()
      {
       return document;
      }),
      InsertAt:function(parent,pos,node)
      {
       var _,matchValue,matchValue1;
       if(node.parentNode===parent)
        {
         matchValue=node.nextSibling;
         matchValue1=[pos,Unchecked.Equals(matchValue,null)?{
          $:0
         }:{
          $:1,
          $0:matchValue
         }];
         _=matchValue1[0].$==1?matchValue1[1].$==1?matchValue1[0].$0===matchValue1[1].$0:false:matchValue1[1].$==0?true:false;
        }
       else
        {
         _=false;
        }
       return!_?pos.$==1?void parent.insertBefore(node,pos.$0):void parent.appendChild(node):null;
      },
      RemoveAttr:function(el,attrName)
      {
       return el.removeAttribute(attrName);
      },
      RemoveClass:function(element,cl)
      {
       jQuery(element).removeClass(cl);
      },
      RemoveNode:function(parent,el)
      {
       return el.parentNode===parent?void parent.removeChild(el):null;
      },
      SetAttr:function(el,name,value)
      {
       return el.setAttribute(name,value);
      },
      SetProperty:function($target,$name,$value)
      {
       var $0=this,$this=this;
       return $target.setProperty($name,$value);
      },
      SetStyle:function(el,name,value)
      {
       return DomUtility.SetProperty(el.style,name,value);
      }
     },
     DoubleInterpolation:Runtime.Class({
      Interpolate:function(t,x,y)
      {
       return x+t*(y-x);
      }
     }),
     DynamicAttrNode:Runtime.Class({
      GetChangeAnim:function()
      {
       return An.get_Empty();
      },
      GetEnterAnim:function()
      {
       return An.get_Empty();
      },
      GetExitAnim:function()
      {
       return An.get_Empty();
      },
      Sync:function(parent)
      {
       if(this.dirty)
        {
         (this.push.call(null,parent))(this.value);
         this.dirty=false;
         return;
        }
       else
        {
         return null;
        }
      },
      get_Changed:function()
      {
       return this.updates;
      }
     },{
      New:function(view,push)
      {
       var r;
       r=Runtime.New(this,{});
       r.push=push;
       r.value=Abbrev.U();
       r.dirty=true;
       r.updates=View.Map(function(x)
       {
        r.value=x;
        r.dirty=true;
        return;
       },view);
       return r;
      }
     }),
     Easing:Runtime.Class({},{
      Custom:function(f)
      {
       return Runtime.New(Easing,{
        TransformTime:f
       });
      },
      get_CubicInOut:function()
      {
       return Easings.CubicInOut();
      }
     }),
     Easings:{
      CubicInOut:Runtime.Field(function()
      {
       return Runtime.New(Easing,{
        TransformTime:function(t)
        {
         var t2;
         t2=t*t;
         return 3*t2-2*(t2*t);
        }
       });
      })
     },
     Flow:Runtime.Class({},{
      get_Do:function()
      {
       return FlowBuilder.New();
      }
     }),
     Flow1:Runtime.Class({},{
      Bind:function(m,k)
      {
       return{
        Render:function(_var)
        {
         return function(cont)
         {
          return(m.Render.call(null,_var))(function(r)
          {
           return(k(r).Render.call(null,_var))(cont);
          });
         };
        }
       };
      },
      Define:function(f)
      {
       return{
        Render:function(_var)
        {
         return function(cont)
         {
          return Var.Set(_var,f(cont));
         };
        }
       };
      },
      Embed:function(fl)
      {
       var _var;
       _var=Var.Create(Doc.get_Empty());
       (fl.Render.call(null,_var))(function()
       {
       });
       return Doc.EmbedView(_var.get_View());
      },
      Map:function(f,x)
      {
       return{
        Render:function(_var)
        {
         return function(cont)
         {
          return(x.Render.call(null,_var))(function(r)
          {
           return cont(f(r));
          });
         };
        }
       };
      },
      Return:function(x)
      {
       return{
        Render:function()
        {
         return function(cont)
         {
          return cont(x);
         };
        }
       };
      },
      Static:function(doc)
      {
       return{
        Render:function(_var)
        {
         return function(cont)
         {
          Var.Set(_var,doc);
          return cont(null);
         };
        }
       };
      }
     }),
     FlowBuilder:Runtime.Class({
      Bind:function(comp,func)
      {
       return Flow1.Bind(comp,func);
      },
      Return:function(value)
      {
       return Flow1.Return(value);
      },
      ReturnFrom:function(inner)
      {
       return inner;
      }
     },{
      New:function()
      {
       return Runtime.New(this,{});
      }
     }),
     Html:{
      A:function(atr,ch)
      {
       return Elements.A(atr,ch);
      },
      A0:function(ch)
      {
       return Elements.A(Runtime.New(T,{
        $:0
       }),ch);
      },
      Attributes:{
       Accept:Runtime.Field(function()
       {
        return"accept";
       }),
       AcceptCharset:Runtime.Field(function()
       {
        return"accept-charset";
       }),
       Accesskey:Runtime.Field(function()
       {
        return"accesskey";
       }),
       Action:Runtime.Field(function()
       {
        return"action";
       }),
       Align:Runtime.Field(function()
       {
        return"align";
       }),
       Alt:Runtime.Field(function()
       {
        return"alt";
       }),
       Async:Runtime.Field(function()
       {
        return"async";
       }),
       AutoComplete:Runtime.Field(function()
       {
        return"autocomplete";
       }),
       AutoFocus:Runtime.Field(function()
       {
        return"autofocus";
       }),
       AutoPlay:Runtime.Field(function()
       {
        return"autoplay";
       }),
       AutoSave:Runtime.Field(function()
       {
        return"autosave";
       }),
       BgColor:Runtime.Field(function()
       {
        return"bgcolor";
       }),
       Border:Runtime.Field(function()
       {
        return"border";
       }),
       Buffered:Runtime.Field(function()
       {
        return"buffered";
       }),
       Challenge:Runtime.Field(function()
       {
        return"challenge";
       }),
       Charset:Runtime.Field(function()
       {
        return"charset";
       }),
       Checked:Runtime.Field(function()
       {
        return"checked";
       }),
       Cite:Runtime.Field(function()
       {
        return"cite";
       }),
       Class:Runtime.Field(function()
       {
        return"class";
       }),
       Code:Runtime.Field(function()
       {
        return"code";
       }),
       Codebase:Runtime.Field(function()
       {
        return"codebase";
       }),
       ColSpan:Runtime.Field(function()
       {
        return"colspan";
       }),
       Color:Runtime.Field(function()
       {
        return"color";
       }),
       Cols:Runtime.Field(function()
       {
        return"cols";
       }),
       Content:Runtime.Field(function()
       {
        return"content";
       }),
       ContentEditable:Runtime.Field(function()
       {
        return"contenteditable";
       }),
       ContextMenu:Runtime.Field(function()
       {
        return"contextmenu";
       }),
       Controls:Runtime.Field(function()
       {
        return"controls";
       }),
       Coords:Runtime.Field(function()
       {
        return"coords";
       }),
       Datetime:Runtime.Field(function()
       {
        return"datetime";
       }),
       Default:Runtime.Field(function()
       {
        return"default";
       }),
       Defer:Runtime.Field(function()
       {
        return"defer";
       }),
       Dir:Runtime.Field(function()
       {
        return"dir";
       }),
       DirName:Runtime.Field(function()
       {
        return"dirname";
       }),
       Disabled:Runtime.Field(function()
       {
        return"disabled";
       }),
       Download:Runtime.Field(function()
       {
        return"download";
       }),
       Draggable:Runtime.Field(function()
       {
        return"draggable";
       }),
       Dropzone:Runtime.Field(function()
       {
        return"dropzone";
       }),
       EncType:Runtime.Field(function()
       {
        return"enctype";
       }),
       For:Runtime.Field(function()
       {
        return"for";
       }),
       Form:Runtime.Field(function()
       {
        return"form";
       }),
       FormAction:Runtime.Field(function()
       {
        return"formaction";
       }),
       Headers:Runtime.Field(function()
       {
        return"headers";
       }),
       Height:Runtime.Field(function()
       {
        return"height";
       }),
       Hidden:Runtime.Field(function()
       {
        return"hidden";
       }),
       High:Runtime.Field(function()
       {
        return"high";
       }),
       Href:Runtime.Field(function()
       {
        return"href";
       }),
       HrefLang:Runtime.Field(function()
       {
        return"hreflang";
       }),
       HttpEquiv:Runtime.Field(function()
       {
        return"http-equiv";
       }),
       ID:Runtime.Field(function()
       {
        return"id";
       }),
       Icon:Runtime.Field(function()
       {
        return"icon";
       }),
       IsMap:Runtime.Field(function()
       {
        return"ismap";
       }),
       ItemProp:Runtime.Field(function()
       {
        return"itemprop";
       }),
       KeyType:Runtime.Field(function()
       {
        return"keytype";
       }),
       Kind:Runtime.Field(function()
       {
        return"kind";
       }),
       Label:Runtime.Field(function()
       {
        return"label";
       }),
       Lang:Runtime.Field(function()
       {
        return"lang";
       }),
       Language:Runtime.Field(function()
       {
        return"language";
       }),
       List:Runtime.Field(function()
       {
        return"list";
       }),
       Loop:Runtime.Field(function()
       {
        return"loop";
       }),
       Low:Runtime.Field(function()
       {
        return"low";
       }),
       Manifest:Runtime.Field(function()
       {
        return"manifest";
       }),
       Max:Runtime.Field(function()
       {
        return"max";
       }),
       MaxLength:Runtime.Field(function()
       {
        return"maxlength";
       }),
       Media:Runtime.Field(function()
       {
        return"media";
       }),
       Method:Runtime.Field(function()
       {
        return"method";
       }),
       Min:Runtime.Field(function()
       {
        return"min";
       }),
       Multiple:Runtime.Field(function()
       {
        return"multiple";
       }),
       Name:Runtime.Field(function()
       {
        return"name";
       }),
       NoValidate:Runtime.Field(function()
       {
        return"novalidate";
       }),
       Open:Runtime.Field(function()
       {
        return"open";
       }),
       Optimum:Runtime.Field(function()
       {
        return"optimum";
       }),
       Pattern:Runtime.Field(function()
       {
        return"pattern";
       }),
       Ping:Runtime.Field(function()
       {
        return"ping";
       }),
       Placeholder:Runtime.Field(function()
       {
        return"placeholder";
       }),
       Poster:Runtime.Field(function()
       {
        return"poster";
       }),
       Preload:Runtime.Field(function()
       {
        return"preload";
       }),
       PubDate:Runtime.Field(function()
       {
        return"pubdate";
       }),
       RadioGroup:Runtime.Field(function()
       {
        return"radiogroup";
       }),
       Readonly:Runtime.Field(function()
       {
        return"readonly";
       }),
       Rel:Runtime.Field(function()
       {
        return"rel";
       }),
       Required:Runtime.Field(function()
       {
        return"required";
       }),
       Reversed:Runtime.Field(function()
       {
        return"reversed";
       }),
       RowSpan:Runtime.Field(function()
       {
        return"rowspan";
       }),
       Rows:Runtime.Field(function()
       {
        return"rows";
       }),
       Sandbox:Runtime.Field(function()
       {
        return"sandbox";
       }),
       Scope:Runtime.Field(function()
       {
        return"scope";
       }),
       Scoped:Runtime.Field(function()
       {
        return"scoped";
       }),
       Seamless:Runtime.Field(function()
       {
        return"seamless";
       }),
       Selected:Runtime.Field(function()
       {
        return"selected";
       }),
       Shape:Runtime.Field(function()
       {
        return"shape";
       }),
       Size:Runtime.Field(function()
       {
        return"size";
       }),
       Sizes:Runtime.Field(function()
       {
        return"sizes";
       }),
       Span:Runtime.Field(function()
       {
        return"span";
       }),
       Spellcheck:Runtime.Field(function()
       {
        return"spellcheck";
       }),
       Src:Runtime.Field(function()
       {
        return"src";
       }),
       SrcLang:Runtime.Field(function()
       {
        return"srclang";
       }),
       Srcdoc:Runtime.Field(function()
       {
        return"srcdoc";
       }),
       Start:Runtime.Field(function()
       {
        return"start";
       }),
       Step:Runtime.Field(function()
       {
        return"step";
       }),
       Style:Runtime.Field(function()
       {
        return"style";
       }),
       Summary:Runtime.Field(function()
       {
        return"summary";
       }),
       TabIndex:Runtime.Field(function()
       {
        return"tabindex";
       }),
       Target:Runtime.Field(function()
       {
        return"target";
       }),
       Title:Runtime.Field(function()
       {
        return"title";
       }),
       Type:Runtime.Field(function()
       {
        return"type";
       }),
       Usemap:Runtime.Field(function()
       {
        return"usemap";
       }),
       Value:Runtime.Field(function()
       {
        return"value";
       }),
       Width:Runtime.Field(function()
       {
        return"width";
       }),
       Wrap:Runtime.Field(function()
       {
        return"wrap";
       })
      },
      Del:function(atr,ch)
      {
       return Elements.Del(atr,ch);
      },
      Del0:function(ch)
      {
       return Elements.Del(Runtime.New(T,{
        $:0
       }),ch);
      },
      Div:function(atr,ch)
      {
       return Elements.Div(atr,ch);
      },
      Div0:function(ch)
      {
       return Elements.Div(Runtime.New(T,{
        $:0
       }),ch);
      },
      Elements:{
       A:function(ats,ch)
       {
        return Doc.Element("a",ats,ch);
       },
       Abbr:function(ats,ch)
       {
        return Doc.Element("abbr",ats,ch);
       },
       Address:function(ats,ch)
       {
        return Doc.Element("address",ats,ch);
       },
       Area:function(ats,ch)
       {
        return Doc.Element("area",ats,ch);
       },
       Article:function(ats,ch)
       {
        return Doc.Element("article",ats,ch);
       },
       Aside:function(ats,ch)
       {
        return Doc.Element("aside",ats,ch);
       },
       Audio:function(ats,ch)
       {
        return Doc.Element("audio",ats,ch);
       },
       B:function(ats,ch)
       {
        return Doc.Element("b",ats,ch);
       },
       BDI:function(ats,ch)
       {
        return Doc.Element("bdi",ats,ch);
       },
       BDO:function(ats,ch)
       {
        return Doc.Element("bdo",ats,ch);
       },
       Base:function(ats,ch)
       {
        return Doc.Element("base",ats,ch);
       },
       BlockQuote:function(ats,ch)
       {
        return Doc.Element("blockquote",ats,ch);
       },
       Body:function(ats,ch)
       {
        return Doc.Element("body",ats,ch);
       },
       Br:function(ats,ch)
       {
        return Doc.Element("br",ats,ch);
       },
       Button:function(ats,ch)
       {
        return Doc.Element("button",ats,ch);
       },
       Canvas:function(ats,ch)
       {
        return Doc.Element("canvas",ats,ch);
       },
       Caption:function(ats,ch)
       {
        return Doc.Element("caption",ats,ch);
       },
       Cite:function(ats,ch)
       {
        return Doc.Element("cite",ats,ch);
       },
       Code:function(ats,ch)
       {
        return Doc.Element("code",ats,ch);
       },
       Col:function(ats,ch)
       {
        return Doc.Element("col",ats,ch);
       },
       ColGroup:function(ats,ch)
       {
        return Doc.Element("colgroup",ats,ch);
       },
       DD:function(ats,ch)
       {
        return Doc.Element("dd",ats,ch);
       },
       DFN:function(ats,ch)
       {
        return Doc.Element("dfn",ats,ch);
       },
       DL:function(ats,ch)
       {
        return Doc.Element("dl",ats,ch);
       },
       DT:function(ats,ch)
       {
        return Doc.Element("dt",ats,ch);
       },
       Data:function(ats,ch)
       {
        return Doc.Element("data",ats,ch);
       },
       DataList:function(ats,ch)
       {
        return Doc.Element("datalist",ats,ch);
       },
       Del:function(ats,ch)
       {
        return Doc.Element("del",ats,ch);
       },
       Details:function(ats,ch)
       {
        return Doc.Element("details",ats,ch);
       },
       Div:function(ats,ch)
       {
        return Doc.Element("div",ats,ch);
       },
       Em:function(ats,ch)
       {
        return Doc.Element("em",ats,ch);
       },
       Embed:function(ats,ch)
       {
        return Doc.Element("embed",ats,ch);
       },
       FieldSet:function(ats,ch)
       {
        return Doc.Element("fieldset",ats,ch);
       },
       FigCaption:function(ats,ch)
       {
        return Doc.Element("figcaption",ats,ch);
       },
       Figure:function(ats,ch)
       {
        return Doc.Element("figure",ats,ch);
       },
       Footer:function(ats,ch)
       {
        return Doc.Element("footer",ats,ch);
       },
       Form:function(ats,ch)
       {
        return Doc.Element("form",ats,ch);
       },
       H1:function(ats,ch)
       {
        return Doc.Element("h1",ats,ch);
       },
       H2:function(ats,ch)
       {
        return Doc.Element("h2",ats,ch);
       },
       H3:function(ats,ch)
       {
        return Doc.Element("h3",ats,ch);
       },
       H4:function(ats,ch)
       {
        return Doc.Element("h4",ats,ch);
       },
       H5:function(ats,ch)
       {
        return Doc.Element("h5",ats,ch);
       },
       H6:function(ats,ch)
       {
        return Doc.Element("h6",ats,ch);
       },
       HR:function(ats,ch)
       {
        return Doc.Element("hr",ats,ch);
       },
       Head:function(ats,ch)
       {
        return Doc.Element("head",ats,ch);
       },
       Header:function(ats,ch)
       {
        return Doc.Element("header",ats,ch);
       },
       Html:function(ats,ch)
       {
        return Doc.Element("html",ats,ch);
       },
       I:function(ats,ch)
       {
        return Doc.Element("i",ats,ch);
       },
       IFrame:function(ats,ch)
       {
        return Doc.Element("iframe",ats,ch);
       },
       Img:function(ats,ch)
       {
        return Doc.Element("img",ats,ch);
       },
       Input:function(ats,ch)
       {
        return Doc.Element("input",ats,ch);
       },
       Ins:function(ats,ch)
       {
        return Doc.Element("ins",ats,ch);
       },
       Kbd:function(ats,ch)
       {
        return Doc.Element("kbd",ats,ch);
       },
       Keygen:function(ats,ch)
       {
        return Doc.Element("keygen",ats,ch);
       },
       LI:function(ats,ch)
       {
        return Doc.Element("li",ats,ch);
       },
       Label:function(ats,ch)
       {
        return Doc.Element("label",ats,ch);
       },
       Legend:function(ats,ch)
       {
        return Doc.Element("legend",ats,ch);
       },
       Link:function(ats,ch)
       {
        return Doc.Element("link",ats,ch);
       },
       Main:function(ats,ch)
       {
        return Doc.Element("main",ats,ch);
       },
       Map:function(ats,ch)
       {
        return Doc.Element("map",ats,ch);
       },
       Mark:function(ats,ch)
       {
        return Doc.Element("mark",ats,ch);
       },
       Menu:function(ats,ch)
       {
        return Doc.Element("menu",ats,ch);
       },
       MenuItem:function(ats,ch)
       {
        return Doc.Element("menuitem",ats,ch);
       },
       Meta:function(ats,ch)
       {
        return Doc.Element("meta",ats,ch);
       },
       Meter:function(ats,ch)
       {
        return Doc.Element("meter",ats,ch);
       },
       Nav:function(ats,ch)
       {
        return Doc.Element("nav",ats,ch);
       },
       NoScript:function(ats,ch)
       {
        return Doc.Element("noscript",ats,ch);
       },
       OL:function(ats,ch)
       {
        return Doc.Element("ol",ats,ch);
       },
       Object:function(ats,ch)
       {
        return Doc.Element("object",ats,ch);
       },
       OptGroup:function(ats,ch)
       {
        return Doc.Element("optgroup",ats,ch);
       },
       Option:function(ats,ch)
       {
        return Doc.Element("option",ats,ch);
       },
       Output:function(ats,ch)
       {
        return Doc.Element("output",ats,ch);
       },
       P:function(ats,ch)
       {
        return Doc.Element("p",ats,ch);
       },
       Param:function(ats,ch)
       {
        return Doc.Element("param",ats,ch);
       },
       Picture:function(ats,ch)
       {
        return Doc.Element("picture",ats,ch);
       },
       Pre:function(ats,ch)
       {
        return Doc.Element("pre",ats,ch);
       },
       Progress:function(ats,ch)
       {
        return Doc.Element("progress",ats,ch);
       },
       Q:function(ats,ch)
       {
        return Doc.Element("q",ats,ch);
       },
       RP:function(ats,ch)
       {
        return Doc.Element("rp",ats,ch);
       },
       RT:function(ats,ch)
       {
        return Doc.Element("rt",ats,ch);
       },
       Ruby:function(ats,ch)
       {
        return Doc.Element("ruby",ats,ch);
       },
       S:function(ats,ch)
       {
        return Doc.Element("s",ats,ch);
       },
       Samp:function(ats,ch)
       {
        return Doc.Element("samp",ats,ch);
       },
       Script:function(ats,ch)
       {
        return Doc.Element("script",ats,ch);
       },
       Section:function(ats,ch)
       {
        return Doc.Element("section",ats,ch);
       },
       Select:function(ats,ch)
       {
        return Doc.Element("select",ats,ch);
       },
       Small:function(ats,ch)
       {
        return Doc.Element("small",ats,ch);
       },
       Source:function(ats,ch)
       {
        return Doc.Element("source",ats,ch);
       },
       Span:function(ats,ch)
       {
        return Doc.Element("span",ats,ch);
       },
       Strong:function(ats,ch)
       {
        return Doc.Element("strong",ats,ch);
       },
       Style:function(ats,ch)
       {
        return Doc.Element("style",ats,ch);
       },
       Sub:function(ats,ch)
       {
        return Doc.Element("sub",ats,ch);
       },
       Summary:function(ats,ch)
       {
        return Doc.Element("summary",ats,ch);
       },
       Sup:function(ats,ch)
       {
        return Doc.Element("sup",ats,ch);
       },
       TBody:function(ats,ch)
       {
        return Doc.Element("tbody",ats,ch);
       },
       TD:function(ats,ch)
       {
        return Doc.Element("td",ats,ch);
       },
       TFoot:function(ats,ch)
       {
        return Doc.Element("tfoot",ats,ch);
       },
       TH:function(ats,ch)
       {
        return Doc.Element("th",ats,ch);
       },
       THead:function(ats,ch)
       {
        return Doc.Element("thead",ats,ch);
       },
       TR:function(ats,ch)
       {
        return Doc.Element("tr",ats,ch);
       },
       Table:function(ats,ch)
       {
        return Doc.Element("table",ats,ch);
       },
       TextArea:function(ats,ch)
       {
        return Doc.Element("textarea",ats,ch);
       },
       Time:function(ats,ch)
       {
        return Doc.Element("time",ats,ch);
       },
       Title:function(ats,ch)
       {
        return Doc.Element("title",ats,ch);
       },
       Track:function(ats,ch)
       {
        return Doc.Element("track",ats,ch);
       },
       U:function(ats,ch)
       {
        return Doc.Element("u",ats,ch);
       },
       UL:function(ats,ch)
       {
        return Doc.Element("ul",ats,ch);
       },
       Var:function(ats,ch)
       {
        return Doc.Element("var",ats,ch);
       },
       Video:function(ats,ch)
       {
        return Doc.Element("video",ats,ch);
       },
       WBR:function(ats,ch)
       {
        return Doc.Element("wbr",ats,ch);
       }
      },
      Form:function(atr,ch)
      {
       return Elements.Form(atr,ch);
      },
      Form0:function(ch)
      {
       return Elements.Form(Runtime.New(T,{
        $:0
       }),ch);
      },
      H1:function(atr,ch)
      {
       return Elements.H1(atr,ch);
      },
      H10:function(ch)
      {
       return Elements.H1(Runtime.New(T,{
        $:0
       }),ch);
      },
      H2:function(atr,ch)
      {
       return Elements.H2(atr,ch);
      },
      H20:function(ch)
      {
       return Elements.H2(Runtime.New(T,{
        $:0
       }),ch);
      },
      H3:function(atr,ch)
      {
       return Elements.H3(atr,ch);
      },
      H30:function(ch)
      {
       return Elements.H3(Runtime.New(T,{
        $:0
       }),ch);
      },
      H4:function(atr,ch)
      {
       return Elements.H4(atr,ch);
      },
      H40:function(ch)
      {
       return Elements.H4(Runtime.New(T,{
        $:0
       }),ch);
      },
      H5:function(atr,ch)
      {
       return Elements.H5(atr,ch);
      },
      H50:function(ch)
      {
       return Elements.H5(Runtime.New(T,{
        $:0
       }),ch);
      },
      H6:function(atr,ch)
      {
       return Elements.H6(atr,ch);
      },
      H60:function(ch)
      {
       return Elements.H6(Runtime.New(T,{
        $:0
       }),ch);
      },
      LI:function(atr,ch)
      {
       return Elements.LI(atr,ch);
      },
      LI0:function(ch)
      {
       return Elements.LI(Runtime.New(T,{
        $:0
       }),ch);
      },
      Label:function(atr,ch)
      {
       return Elements.Label(atr,ch);
      },
      Label0:function(ch)
      {
       return Elements.Label(Runtime.New(T,{
        $:0
       }),ch);
      },
      Nav:function(atr,ch)
      {
       return Elements.Nav(atr,ch);
      },
      Nav0:function(ch)
      {
       return Elements.Nav(Runtime.New(T,{
        $:0
       }),ch);
      },
      OL:function(atr,ch)
      {
       return Elements.OL(atr,ch);
      },
      OL0:function(ch)
      {
       return Elements.OL(Runtime.New(T,{
        $:0
       }),ch);
      },
      P:function(atr,ch)
      {
       return Elements.P(atr,ch);
      },
      P0:function(ch)
      {
       return Elements.P(Runtime.New(T,{
        $:0
       }),ch);
      },
      Span:function(atr,ch)
      {
       return Elements.Span(atr,ch);
      },
      Span0:function(ch)
      {
       return Elements.Span(Runtime.New(T,{
        $:0
       }),ch);
      },
      SvgAttributes:{
       AccentHeight:Runtime.Field(function()
       {
        return"accent-height";
       }),
       Accumulate:Runtime.Field(function()
       {
        return"accumulate";
       }),
       Additive:Runtime.Field(function()
       {
        return"additive";
       }),
       AlignmentBaseline:Runtime.Field(function()
       {
        return"alignment-baseline";
       }),
       Ascent:Runtime.Field(function()
       {
        return"ascent";
       }),
       AttributeName:Runtime.Field(function()
       {
        return"attributeName";
       }),
       AttributeType:Runtime.Field(function()
       {
        return"attributeType";
       }),
       Azimuth:Runtime.Field(function()
       {
        return"azimuth";
       }),
       BaseFrequency:Runtime.Field(function()
       {
        return"baseFrequency";
       }),
       BaselineShift:Runtime.Field(function()
       {
        return"baseline-shift";
       }),
       Begin:Runtime.Field(function()
       {
        return"begin";
       }),
       Bias:Runtime.Field(function()
       {
        return"bias";
       }),
       CX:Runtime.Field(function()
       {
        return"cx";
       }),
       CY:Runtime.Field(function()
       {
        return"cy";
       }),
       CalcMode:Runtime.Field(function()
       {
        return"calcMode";
       }),
       Class:Runtime.Field(function()
       {
        return"class";
       }),
       Clip:Runtime.Field(function()
       {
        return"clip";
       }),
       ClipPath:Runtime.Field(function()
       {
        return"clip-path";
       }),
       ClipPathUnits:Runtime.Field(function()
       {
        return"clipPathUnits";
       }),
       ClipRule:Runtime.Field(function()
       {
        return"clip-rule";
       }),
       Color:Runtime.Field(function()
       {
        return"color";
       }),
       ColorInterpolation:Runtime.Field(function()
       {
        return"color-interpolation";
       }),
       ColorInterpolationFilters:Runtime.Field(function()
       {
        return"color-interpolation-filters";
       }),
       ColorProfile:Runtime.Field(function()
       {
        return"color-profile";
       }),
       ColorRendering:Runtime.Field(function()
       {
        return"color-rendering";
       }),
       ContentScriptType:Runtime.Field(function()
       {
        return"contentScriptType";
       }),
       ContentStyleType:Runtime.Field(function()
       {
        return"contentStyleType";
       }),
       Cursor:Runtime.Field(function()
       {
        return"cursor";
       }),
       D:Runtime.Field(function()
       {
        return"d";
       }),
       DX:Runtime.Field(function()
       {
        return"dx";
       }),
       DY:Runtime.Field(function()
       {
        return"dy";
       }),
       DiffuseConstant:Runtime.Field(function()
       {
        return"diffuseConstant";
       }),
       Direction:Runtime.Field(function()
       {
        return"direction";
       }),
       Display:Runtime.Field(function()
       {
        return"display";
       }),
       Divisor:Runtime.Field(function()
       {
        return"divisor";
       }),
       DominantBaseline:Runtime.Field(function()
       {
        return"dominant-baseline";
       }),
       Dur:Runtime.Field(function()
       {
        return"dur";
       }),
       EdgeMode:Runtime.Field(function()
       {
        return"edgeMode";
       }),
       Elevation:Runtime.Field(function()
       {
        return"elevation";
       }),
       End:Runtime.Field(function()
       {
        return"end";
       }),
       ExternalResourcesRequired:Runtime.Field(function()
       {
        return"externalResourcesRequired";
       }),
       Fill:Runtime.Field(function()
       {
        return"fill";
       }),
       FillOpacity:Runtime.Field(function()
       {
        return"fill-opacity";
       }),
       FillRule:Runtime.Field(function()
       {
        return"fill-rule";
       }),
       Filter:Runtime.Field(function()
       {
        return"filter";
       }),
       FilterRes:Runtime.Field(function()
       {
        return"filterRes";
       }),
       FilterUnits:Runtime.Field(function()
       {
        return"filterUnits";
       }),
       FloodColor:Runtime.Field(function()
       {
        return"flood-color";
       }),
       FloodOpacity:Runtime.Field(function()
       {
        return"flood-opacity";
       }),
       FontFamily:Runtime.Field(function()
       {
        return"font-family";
       }),
       FontSize:Runtime.Field(function()
       {
        return"font-size";
       }),
       FontSizeAdjust:Runtime.Field(function()
       {
        return"font-size-adjust";
       }),
       FontStretch:Runtime.Field(function()
       {
        return"font-stretch";
       }),
       FontStyle:Runtime.Field(function()
       {
        return"font-style";
       }),
       FontVariant:Runtime.Field(function()
       {
        return"font-variant";
       }),
       FontWeight:Runtime.Field(function()
       {
        return"font-weight";
       }),
       From:Runtime.Field(function()
       {
        return"from";
       }),
       GradientTransform:Runtime.Field(function()
       {
        return"gradientTransform";
       }),
       GradientUnits:Runtime.Field(function()
       {
        return"gradientUnits";
       }),
       Height:Runtime.Field(function()
       {
        return"height";
       }),
       IN:Runtime.Field(function()
       {
        return"in";
       }),
       ImageRendering:Runtime.Field(function()
       {
        return"image-rendering";
       }),
       In2:Runtime.Field(function()
       {
        return"in2";
       }),
       K1:Runtime.Field(function()
       {
        return"k1";
       }),
       K2:Runtime.Field(function()
       {
        return"k2";
       }),
       K3:Runtime.Field(function()
       {
        return"k3";
       }),
       K4:Runtime.Field(function()
       {
        return"k4";
       }),
       KernelMatrix:Runtime.Field(function()
       {
        return"kernelMatrix";
       }),
       KernelUnitLength:Runtime.Field(function()
       {
        return"kernelUnitLength";
       }),
       Kerning:Runtime.Field(function()
       {
        return"kerning";
       }),
       KeySplines:Runtime.Field(function()
       {
        return"keySplines";
       }),
       KeyTimes:Runtime.Field(function()
       {
        return"keyTimes";
       }),
       LetterSpacing:Runtime.Field(function()
       {
        return"letter-spacing";
       }),
       LightingColor:Runtime.Field(function()
       {
        return"lighting-color";
       }),
       LimitingConeAngle:Runtime.Field(function()
       {
        return"limitingConeAngle";
       }),
       Local:Runtime.Field(function()
       {
        return"local";
       }),
       MarkerEnd:Runtime.Field(function()
       {
        return"marker-end";
       }),
       MarkerHeight:Runtime.Field(function()
       {
        return"markerHeight";
       }),
       MarkerMid:Runtime.Field(function()
       {
        return"marker-mid";
       }),
       MarkerStart:Runtime.Field(function()
       {
        return"marker-start";
       }),
       MarkerUnits:Runtime.Field(function()
       {
        return"markerUnits";
       }),
       MarkerWidth:Runtime.Field(function()
       {
        return"markerWidth";
       }),
       Mask:Runtime.Field(function()
       {
        return"mask";
       }),
       MaskContentUnits:Runtime.Field(function()
       {
        return"maskContentUnits";
       }),
       MaskUnits:Runtime.Field(function()
       {
        return"maskUnits";
       }),
       Max:Runtime.Field(function()
       {
        return"max";
       }),
       Min:Runtime.Field(function()
       {
        return"min";
       }),
       Mode:Runtime.Field(function()
       {
        return"mode";
       }),
       NumOctaves:Runtime.Field(function()
       {
        return"numOctaves";
       }),
       Opacity:Runtime.Field(function()
       {
        return"opacity";
       }),
       Operator:Runtime.Field(function()
       {
        return"operator";
       }),
       Order:Runtime.Field(function()
       {
        return"order";
       }),
       Overflow:Runtime.Field(function()
       {
        return"overflow";
       }),
       PaintOrder:Runtime.Field(function()
       {
        return"paint-order";
       }),
       PathLength:Runtime.Field(function()
       {
        return"pathLength";
       }),
       PatternContentUnits:Runtime.Field(function()
       {
        return"patternContentUnits";
       }),
       PatternTransform:Runtime.Field(function()
       {
        return"patternTransform";
       }),
       PatternUnits:Runtime.Field(function()
       {
        return"patternUnits";
       }),
       PointerEvents:Runtime.Field(function()
       {
        return"pointer-events";
       }),
       Points:Runtime.Field(function()
       {
        return"points";
       }),
       PointsAtX:Runtime.Field(function()
       {
        return"pointsAtX";
       }),
       PointsAtY:Runtime.Field(function()
       {
        return"pointsAtY";
       }),
       PointsAtZ:Runtime.Field(function()
       {
        return"pointsAtZ";
       }),
       PreserveAlpha:Runtime.Field(function()
       {
        return"preserveAlpha";
       }),
       PreserveAspectRatio:Runtime.Field(function()
       {
        return"preserveAspectRatio";
       }),
       PrimitiveUnits:Runtime.Field(function()
       {
        return"primitiveUnits";
       }),
       R:Runtime.Field(function()
       {
        return"r";
       }),
       RX:Runtime.Field(function()
       {
        return"rx";
       }),
       RY:Runtime.Field(function()
       {
        return"ry";
       }),
       Radius:Runtime.Field(function()
       {
        return"radius";
       }),
       RepeatCount:Runtime.Field(function()
       {
        return"repeatCount";
       }),
       RepeatDur:Runtime.Field(function()
       {
        return"repeatDur";
       }),
       RequiredFeatures:Runtime.Field(function()
       {
        return"requiredFeatures";
       }),
       Restart:Runtime.Field(function()
       {
        return"restart";
       }),
       Result:Runtime.Field(function()
       {
        return"result";
       }),
       Scale:Runtime.Field(function()
       {
        return"scale";
       }),
       Seed:Runtime.Field(function()
       {
        return"seed";
       }),
       ShapeRendering:Runtime.Field(function()
       {
        return"shape-rendering";
       }),
       SpecularConstant:Runtime.Field(function()
       {
        return"specularConstant";
       }),
       SpecularExponent:Runtime.Field(function()
       {
        return"specularExponent";
       }),
       StdDeviation:Runtime.Field(function()
       {
        return"stdDeviation";
       }),
       StitchTiles:Runtime.Field(function()
       {
        return"stitchTiles";
       }),
       StopColor:Runtime.Field(function()
       {
        return"stop-color";
       }),
       StopOpacity:Runtime.Field(function()
       {
        return"stop-opacity";
       }),
       Stroke:Runtime.Field(function()
       {
        return"stroke";
       }),
       StrokeDashArray:Runtime.Field(function()
       {
        return"stroke-dasharray";
       }),
       StrokeDashOffset:Runtime.Field(function()
       {
        return"stroke-dashoffset";
       }),
       StrokeLineCap:Runtime.Field(function()
       {
        return"stroke-linecap";
       }),
       StrokeLineJoin:Runtime.Field(function()
       {
        return"stroke-linejoin";
       }),
       StrokeMiterLimit:Runtime.Field(function()
       {
        return"stroke-miterlimit";
       }),
       StrokeOpacity:Runtime.Field(function()
       {
        return"stroke-opacity";
       }),
       StrokeWidth:Runtime.Field(function()
       {
        return"stroke-width";
       }),
       Style:Runtime.Field(function()
       {
        return"style";
       }),
       SurfaceScale:Runtime.Field(function()
       {
        return"surfaceScale";
       }),
       TargetX:Runtime.Field(function()
       {
        return"targetX";
       }),
       TargetY:Runtime.Field(function()
       {
        return"targetY";
       }),
       TextAnchor:Runtime.Field(function()
       {
        return"text-anchor";
       }),
       TextDecoration:Runtime.Field(function()
       {
        return"text-decoration";
       }),
       TextRendering:Runtime.Field(function()
       {
        return"text-rendering";
       }),
       To:Runtime.Field(function()
       {
        return"to";
       }),
       Transform:Runtime.Field(function()
       {
        return"transform";
       }),
       Type:Runtime.Field(function()
       {
        return"type";
       }),
       Values:Runtime.Field(function()
       {
        return"values";
       }),
       ViewBox:Runtime.Field(function()
       {
        return"viewBox";
       }),
       Visibility:Runtime.Field(function()
       {
        return"visibility";
       }),
       Width:Runtime.Field(function()
       {
        return"width";
       }),
       WordSpacing:Runtime.Field(function()
       {
        return"word-spacing";
       }),
       WritingMode:Runtime.Field(function()
       {
        return"writing-mode";
       }),
       X:Runtime.Field(function()
       {
        return"x";
       }),
       X1:Runtime.Field(function()
       {
        return"x1";
       }),
       X2:Runtime.Field(function()
       {
        return"x2";
       }),
       XChannelSelector:Runtime.Field(function()
       {
        return"xChannelSelector";
       }),
       Y:Runtime.Field(function()
       {
        return"y";
       }),
       Y1:Runtime.Field(function()
       {
        return"y1";
       }),
       Y2:Runtime.Field(function()
       {
        return"y2";
       }),
       YChannelSelector:Runtime.Field(function()
       {
        return"yChannelSelector";
       }),
       Z:Runtime.Field(function()
       {
        return"z";
       })
      },
      SvgElements:{
       A:function(ats,ch)
       {
        return Doc.SvgElement("a",ats,ch);
       },
       AltGlyph:function(ats,ch)
       {
        return Doc.SvgElement("altglyph",ats,ch);
       },
       AltGlyphDef:function(ats,ch)
       {
        return Doc.SvgElement("altglyphdef",ats,ch);
       },
       AltGlyphItem:function(ats,ch)
       {
        return Doc.SvgElement("altglyphitem",ats,ch);
       },
       Animate:function(ats,ch)
       {
        return Doc.SvgElement("animate",ats,ch);
       },
       AnimateColor:function(ats,ch)
       {
        return Doc.SvgElement("animatecolor",ats,ch);
       },
       AnimateMotion:function(ats,ch)
       {
        return Doc.SvgElement("animatemotion",ats,ch);
       },
       AnimateTransform:function(ats,ch)
       {
        return Doc.SvgElement("animatetransform",ats,ch);
       },
       Circle:function(ats,ch)
       {
        return Doc.SvgElement("circle",ats,ch);
       },
       ClipPath:function(ats,ch)
       {
        return Doc.SvgElement("clippath",ats,ch);
       },
       ColorProfile:function(ats,ch)
       {
        return Doc.SvgElement("color-profile",ats,ch);
       },
       Cursor:function(ats,ch)
       {
        return Doc.SvgElement("cursor",ats,ch);
       },
       Defs:function(ats,ch)
       {
        return Doc.SvgElement("defs",ats,ch);
       },
       Desc:function(ats,ch)
       {
        return Doc.SvgElement("desc",ats,ch);
       },
       Ellipse:function(ats,ch)
       {
        return Doc.SvgElement("ellipse",ats,ch);
       },
       FeBlend:function(ats,ch)
       {
        return Doc.SvgElement("feblend",ats,ch);
       },
       FeColorMatrix:function(ats,ch)
       {
        return Doc.SvgElement("fecolormatrix",ats,ch);
       },
       FeComponentTransfer:function(ats,ch)
       {
        return Doc.SvgElement("fecomponenttransfer",ats,ch);
       },
       FeComposite:function(ats,ch)
       {
        return Doc.SvgElement("fecomposite",ats,ch);
       },
       FeConvolveMatrix:function(ats,ch)
       {
        return Doc.SvgElement("feconvolvematrix",ats,ch);
       },
       FeDiffuseLighting:function(ats,ch)
       {
        return Doc.SvgElement("fediffuselighting",ats,ch);
       },
       FeDisplacementMap:function(ats,ch)
       {
        return Doc.SvgElement("fedisplacementmap",ats,ch);
       },
       FeDistantLight:function(ats,ch)
       {
        return Doc.SvgElement("fedistantlight",ats,ch);
       },
       FeFlood:function(ats,ch)
       {
        return Doc.SvgElement("feflood",ats,ch);
       },
       FeFuncA:function(ats,ch)
       {
        return Doc.SvgElement("fefunca",ats,ch);
       },
       FeFuncB:function(ats,ch)
       {
        return Doc.SvgElement("fefuncb",ats,ch);
       },
       FeFuncG:function(ats,ch)
       {
        return Doc.SvgElement("fefuncg",ats,ch);
       },
       FeFuncR:function(ats,ch)
       {
        return Doc.SvgElement("fefuncr",ats,ch);
       },
       FeGaussianBlur:function(ats,ch)
       {
        return Doc.SvgElement("fegaussianblur",ats,ch);
       },
       FeImage:function(ats,ch)
       {
        return Doc.SvgElement("feimage",ats,ch);
       },
       FeMerge:function(ats,ch)
       {
        return Doc.SvgElement("femerge",ats,ch);
       },
       FeMergeNode:function(ats,ch)
       {
        return Doc.SvgElement("femergenode",ats,ch);
       },
       FeMorphology:function(ats,ch)
       {
        return Doc.SvgElement("femorphology",ats,ch);
       },
       FeOffset:function(ats,ch)
       {
        return Doc.SvgElement("feoffset",ats,ch);
       },
       FePointLight:function(ats,ch)
       {
        return Doc.SvgElement("fepointlight",ats,ch);
       },
       FeSpecularLighting:function(ats,ch)
       {
        return Doc.SvgElement("fespecularlighting",ats,ch);
       },
       FeSpotLight:function(ats,ch)
       {
        return Doc.SvgElement("fespotlight",ats,ch);
       },
       FeTile:function(ats,ch)
       {
        return Doc.SvgElement("fetile",ats,ch);
       },
       FeTurbulence:function(ats,ch)
       {
        return Doc.SvgElement("feturbulence",ats,ch);
       },
       Filter:function(ats,ch)
       {
        return Doc.SvgElement("filter",ats,ch);
       },
       Font:function(ats,ch)
       {
        return Doc.SvgElement("font",ats,ch);
       },
       FontFace:function(ats,ch)
       {
        return Doc.SvgElement("font-face",ats,ch);
       },
       FontFaceFormat:function(ats,ch)
       {
        return Doc.SvgElement("font-face-format",ats,ch);
       },
       FontFaceName:function(ats,ch)
       {
        return Doc.SvgElement("font-face-name",ats,ch);
       },
       FontFaceSrc:function(ats,ch)
       {
        return Doc.SvgElement("font-face-src",ats,ch);
       },
       FontFaceUri:function(ats,ch)
       {
        return Doc.SvgElement("font-face-uri",ats,ch);
       },
       ForeignObject:function(ats,ch)
       {
        return Doc.SvgElement("foreignobject",ats,ch);
       },
       G:function(ats,ch)
       {
        return Doc.SvgElement("g",ats,ch);
       },
       Glyph:function(ats,ch)
       {
        return Doc.SvgElement("glyph",ats,ch);
       },
       GlyphRef:function(ats,ch)
       {
        return Doc.SvgElement("glyphref",ats,ch);
       },
       HKern:function(ats,ch)
       {
        return Doc.SvgElement("hkern",ats,ch);
       },
       Image:function(ats,ch)
       {
        return Doc.SvgElement("image",ats,ch);
       },
       Line:function(ats,ch)
       {
        return Doc.SvgElement("line",ats,ch);
       },
       LinearGradient:function(ats,ch)
       {
        return Doc.SvgElement("lineargradient",ats,ch);
       },
       MPath:function(ats,ch)
       {
        return Doc.SvgElement("mpath",ats,ch);
       },
       Marker:function(ats,ch)
       {
        return Doc.SvgElement("marker",ats,ch);
       },
       Mask:function(ats,ch)
       {
        return Doc.SvgElement("mask",ats,ch);
       },
       Metadata:function(ats,ch)
       {
        return Doc.SvgElement("metadata",ats,ch);
       },
       MissingGlyph:function(ats,ch)
       {
        return Doc.SvgElement("missing-glyph",ats,ch);
       },
       Path:function(ats,ch)
       {
        return Doc.SvgElement("path",ats,ch);
       },
       Pattern:function(ats,ch)
       {
        return Doc.SvgElement("pattern",ats,ch);
       },
       Polygon:function(ats,ch)
       {
        return Doc.SvgElement("polygon",ats,ch);
       },
       Polyline:function(ats,ch)
       {
        return Doc.SvgElement("polyline",ats,ch);
       },
       RadialGradient:function(ats,ch)
       {
        return Doc.SvgElement("radialgradient",ats,ch);
       },
       Rect:function(ats,ch)
       {
        return Doc.SvgElement("rect",ats,ch);
       },
       Script:function(ats,ch)
       {
        return Doc.SvgElement("script",ats,ch);
       },
       Set:function(ats,ch)
       {
        return Doc.SvgElement("set",ats,ch);
       },
       Stop:function(ats,ch)
       {
        return Doc.SvgElement("stop",ats,ch);
       },
       Style:function(ats,ch)
       {
        return Doc.SvgElement("style",ats,ch);
       },
       Svg:function(ats,ch)
       {
        return Doc.SvgElement("svg",ats,ch);
       },
       Switch:function(ats,ch)
       {
        return Doc.SvgElement("switch",ats,ch);
       },
       Symbol:function(ats,ch)
       {
        return Doc.SvgElement("symbol",ats,ch);
       },
       TRef:function(ats,ch)
       {
        return Doc.SvgElement("tref",ats,ch);
       },
       TSpan:function(ats,ch)
       {
        return Doc.SvgElement("tspan",ats,ch);
       },
       Text:function(ats,ch)
       {
        return Doc.SvgElement("text",ats,ch);
       },
       TextPath:function(ats,ch)
       {
        return Doc.SvgElement("textpath",ats,ch);
       },
       Title:function(ats,ch)
       {
        return Doc.SvgElement("title",ats,ch);
       },
       Use:function(ats,ch)
       {
        return Doc.SvgElement("use",ats,ch);
       },
       VKern:function(ats,ch)
       {
        return Doc.SvgElement("vkern",ats,ch);
       },
       View:function(ats,ch)
       {
        return Doc.SvgElement("view",ats,ch);
       }
      },
      TBody:function(atr,ch)
      {
       return Elements.TBody(atr,ch);
      },
      TBody0:function(ch)
      {
       return Elements.TBody(Runtime.New(T,{
        $:0
       }),ch);
      },
      TD:function(atr,ch)
      {
       return Elements.TD(atr,ch);
      },
      TD0:function(ch)
      {
       return Elements.TD(Runtime.New(T,{
        $:0
       }),ch);
      },
      THead:function(atr,ch)
      {
       return Elements.THead(atr,ch);
      },
      THead0:function(ch)
      {
       return Elements.THead(Runtime.New(T,{
        $:0
       }),ch);
      },
      TR:function(atr,ch)
      {
       return Elements.TR(atr,ch);
      },
      TR0:function(ch)
      {
       return Elements.TR(Runtime.New(T,{
        $:0
       }),ch);
      },
      Table:function(atr,ch)
      {
       return Elements.Table(atr,ch);
      },
      Table0:function(ch)
      {
       return Elements.Table(Runtime.New(T,{
        $:0
       }),ch);
      },
      UL:function(atr,ch)
      {
       return Elements.UL(atr,ch);
      },
      UL0:function(ch)
      {
       return Elements.UL(Runtime.New(T,{
        $:0
       }),ch);
      }
     },
     Input:{
      ActivateButtonListener:Runtime.Field(function()
      {
       var _buttonListener_39_1,_;
       _buttonListener_39_1=function(evt,down)
       {
        var matchValue;
        matchValue=evt.button;
        return matchValue===0?Var.Set(Input.MouseBtnSt1().Left,down):matchValue===1?Var.Set(Input.MouseBtnSt1().Middle,down):matchValue===2?Var.Set(Input.MouseBtnSt1().Right,down):null;
       };
       if(!Input.MouseBtnSt1().Active)
        {
         Input.MouseBtnSt1().Active=true;
         document.addEventListener("mousedown",function(evt)
         {
          return _buttonListener_39_1(evt,true);
         },false);
         _=document.addEventListener("mouseup",function(evt)
         {
          return _buttonListener_39_1(evt,false);
         },false);
        }
       else
        {
         _=null;
        }
       return _;
      }),
      ActivateKeyListener:Runtime.Field(function()
      {
       var _;
       if(!Input.KeyListenerState().KeyListenerActive)
        {
         jQuery(document).keydown(function(evt)
         {
          var keyCode,xs,p,b;
          keyCode=evt.which;
          Var.Set(Input.KeyListenerState().LastPressed,keyCode);
          xs=Var.Get(Input.KeyListenerState().KeysPressed);
          p=function(x)
          {
           return x===keyCode;
          };
          if(!Seq.exists(p,xs))
           {
            b=List.ofArray([keyCode]);
            return Input.KeyListenerState().KeysPressed.set_Value(List.append(xs,b));
           }
          else
           {
            return null;
           }
         });
         _=void jQuery(document).keyup(function(evt)
         {
          var keyCode,predicate,arg10;
          keyCode=evt.which;
          predicate=function(x)
          {
           return x!==keyCode;
          };
          arg10=function(list)
          {
           return List.filter(predicate,list);
          };
          return Var1.Update(Input.KeyListenerState().KeysPressed,arg10);
         });
        }
       else
        {
         _=null;
        }
       return _;
      }),
      KeyListenerState:Runtime.Field(function()
      {
       return{
        KeysPressed:Var.Create(Runtime.New(T,{
         $:0
        })),
        KeyListenerActive:false,
        LastPressed:Var.Create(-1)
       };
      }),
      Keyboard:Runtime.Class({},{
       IsPressed:function(key)
       {
        var predicate;
        Input.ActivateKeyListener();
        predicate=function(x)
        {
         return x===key;
        };
        return View.Map(function(list)
        {
         return Seq.exists(predicate,list);
        },Input.KeyListenerState().KeysPressed.get_View());
       },
       get_KeysPressed:function()
       {
        Input.ActivateKeyListener();
        return Input.KeyListenerState().KeysPressed.get_View();
       },
       get_LastPressed:function()
       {
        Input.ActivateKeyListener();
        return Input.KeyListenerState().LastPressed.get_View();
       }
      }),
      Mouse:Runtime.Class({},{
       get_LeftPressed:function()
       {
        Input.ActivateButtonListener();
        return Input.MouseBtnSt1().Left.get_View();
       },
       get_MiddlePressed:function()
       {
        Input.ActivateButtonListener();
        return Input.MouseBtnSt1().Middle.get_View();
       },
       get_MousePressed:function()
       {
        Input.ActivateButtonListener();
        return View1.Apply(View1.Apply(View1.Apply(View1.Const(function(l)
        {
         return function(m)
         {
          return function(r)
          {
           return(l?true:m)?true:r;
          };
         };
        }),Input.MouseBtnSt1().Left.get_View()),Input.MouseBtnSt1().Middle.get_View()),Input.MouseBtnSt1().Right.get_View());
       },
       get_Position:function()
       {
        var onMouseMove;
        onMouseMove=function(evt)
        {
         return Var.Set(Input.MousePosSt1().PosV,[evt.clientX,evt.clientY]);
        };
        if(!Input.MousePosSt1().Active)
         {
          document.addEventListener("mousemove",onMouseMove,false);
          Input.MousePosSt1().Active=true;
         }
        return View.FromVar(Input.MousePosSt1().PosV);
       },
       get_RightPressed:function()
       {
        Input.ActivateButtonListener();
        return Input.MouseBtnSt1().Right.get_View();
       }
      }),
      MouseBtnSt1:Runtime.Field(function()
      {
       return{
        Active:false,
        Left:Var.Create(false),
        Middle:Var.Create(false),
        Right:Var.Create(false)
       };
      }),
      MousePosSt1:Runtime.Field(function()
      {
       return{
        Active:false,
        PosV:Var.Create([0,0])
       };
      })
     },
     Interpolation1:Runtime.Class({},{
      get_Double:function()
      {
       return Runtime.New(DoubleInterpolation,{
        $:0
       });
      }
     }),
     Key:Runtime.Class({},{
      Fresh:function()
      {
       return Runtime.New(Key,{
        $:0,
        $0:Fresh.Int()
       });
      }
     }),
     ListModel:Runtime.Class({
      Add:function(item)
      {
       var v,m=this;
       v=this.Var.get_Value();
       if(!ListModels.Contains(this.Key,item,v))
        {
         v.push(item);
         return this.Var.set_Value(v);
        }
       else
        {
         Arrays.set(v,Arrays.findINdex(function(it)
         {
          return Unchecked.Equals(m.Key.call(null,it),m.Key.call(null,item));
         },v),item);
         return m.Var.set_Value(v);
        }
      },
      Clear:function()
      {
       return this.Var.set_Value([]);
      },
      ContainsKey:function(key)
      {
       var f,m=this,arr;
       f=function(it)
       {
        return Unchecked.Equals(m.Key.call(null,it),key);
       };
       arr=m.Var.get_Value();
       return Seq.exists(f,arr);
      },
      FindByKey:function(key)
      {
       var m=this;
       return Arrays.find(function(it)
       {
        return Unchecked.Equals(m.Key.call(null,it),key);
       },m.Var.get_Value());
      },
      Iter:function(fn)
      {
       return Arrays.iter(fn,this.Var.get_Value());
      },
      Remove:function(item)
      {
       var v,keyFn,k;
       v=this.Var.get_Value();
       if(ListModels.Contains(this.Key,item,v))
        {
         keyFn=this.Key;
         k=keyFn(item);
         return this.Var.set_Value(Arrays.filter(function(i)
         {
          return!Unchecked.Equals(keyFn(i),k);
         },v));
        }
       else
        {
         return null;
        }
      },
      RemoveByKey:function(key)
      {
       var m=this;
       return this.Var.set_Value(Arrays.filter(function(i)
       {
        return!Unchecked.Equals(m.Key.call(null,i),key);
       },m.Var.get_Value()));
      },
      Set:function(lst)
      {
       return this.Var.set_Value(Arrays.ofSeq(lst));
      },
      UpdateBy:function(fn,key)
      {
       var v,index,m=this,matchValue;
       v=this.Var.get_Value();
       if(this.ContainsKey(key))
        {
         index=Arrays.findINdex(function(it)
         {
          return Unchecked.Equals(m.Key.call(null,it),key);
         },v);
         matchValue=fn(Arrays.get(v,index));
         if(matchValue.$==1)
          {
           Arrays.set(v,index,matchValue.$0);
           return m.Var.set_Value(v);
          }
         else
          {
           return m.RemoveByKey(key);
          }
        }
       else
        {
         return null;
        }
      }
     },{
      Create:function(key,init)
      {
       var _var;
       _var=Var.Create(Seq.toArray(Seq.distinctBy(key,init)));
       return Runtime.New(ListModel,{
        Key:key,
        Var:_var,
        View:View.Map(function(x)
        {
         return x.slice();
        },_var.get_View())
       });
      },
      FromSeq:function(xs)
      {
       return ListModel.Create(function(x)
       {
        return x;
       },xs);
      }
     }),
     ListModel1:Runtime.Class({},{
      View:function(m)
      {
       return m.View;
      }
     }),
     ListModels:{
      Contains:function(keyFn,item,xs)
      {
       var t,f;
       t=keyFn(item);
       f=function(it)
       {
        return Unchecked.Equals(keyFn(it),t);
       };
       return Seq.exists(f,xs);
      }
     },
     Model:Runtime.Class({
      get_View:function()
      {
       return Model.View(this);
      }
     },{
      Create:function(proj,init)
      {
       var _var;
       _var=Var.Create(init);
       return Runtime.New(Model,{
        $:0,
        $0:_var,
        $1:View.Map(proj,_var.get_View())
       });
      },
      Update:function(update,_arg1)
      {
       return Var1.Update(_arg1.$0,function(x)
       {
        update(x);
        return x;
       });
      },
      View:function(_arg2)
      {
       return _arg2.$1;
      }
     }),
     Route:{
      Append:function(_arg2,_arg1)
      {
       return{
        $:0,
        $0:AppendList1.Append(_arg2.$0,_arg1.$0)
       };
      },
      FromList:function(xs)
      {
       return{
        $:0,
        $0:AppendList1.FromArray(Arrays.ofSeq(xs))
       };
      },
      MakeHash:function(_arg1)
      {
       return Strings.concat("/",Arrays.map(function(x)
       {
        return encodeURIComponent(x);
       },AppendList1.ToArray(_arg1.$0)));
      },
      NoHash:function(s)
      {
       return Strings.StartsWith(s,"#")?s.substring(1):s;
      },
      ParseHash:function(hash)
      {
       var _this,sep;
       _this=Route.NoHash(hash);
       sep=[47];
       return{
        $:0,
        $0:AppendList1.FromArray(Arrays.map(function(x)
        {
         return decodeURIComponent(x);
        },Strings.SplitChars(_this,sep,1)))
       };
      },
      SameHash:function(a,b)
      {
       return Route.NoHash(a)===Route.NoHash(b);
      },
      ToList:function(_arg1)
      {
       var array;
       array=AppendList1.ToArray(_arg1.$0);
       return List.ofArray(array);
      }
     },
     RouteMap1:Runtime.Class({},{
      Create:function(ser,des)
      {
       return{
        Des:des,
        Ser:ser
       };
      },
      Install:function(map)
      {
       return Routing.InstallMap(map);
      }
     }),
     Router:Runtime.Class({},{
      Dir:function(prefix,sites)
      {
       return Router1.Prefix(prefix,Router.Merge(sites));
      },
      Install:function(key,site)
      {
       return Routing.Install(key,site);
      },
      Merge:function(sites)
      {
       return Routing.MergeRouters(sites);
      }
     }),
     Router1:Runtime.Class({},{
      Prefix:function(prefix,_arg1)
      {
       return{
        $:0,
        $0:_arg1.$0,
        $1:Trie1.Prefix(prefix,_arg1.$1)
       };
      },
      Route:function(r,init,render)
      {
       return Routing.DefineRoute(r,init,render);
      }
     }),
     Routing:{
      ComputeBodies:function(trie)
      {
       var d;
       d=Dictionary.New12();
       Arrays.iter(function(body)
       {
        return d.set_Item(body.RouteId,body);
       },Trie1.ToArray(trie));
       return d;
      },
      DefineRoute:function(r,init,render)
      {
       var state,id,site,t;
       state=Var.Create(init);
       id=Fresh.Int();
       site=(render({
        $:0,
        $0:id
       }))(state);
       t=Trie1.Leaf({
        $:0,
        $0:id,
        $1:function(ctx)
        {
         View1.Sink(function(va)
         {
          return ctx.UpdateRoute.call(null,Routing.DoLink(r,va));
         },state.get_View());
         return{
          OnRouteChanged:function(route)
          {
           return state.set_Value(Routing.DoRoute(r,route));
          },
          OnSelect:function()
          {
           return ctx.UpdateRoute.call(null,Routing.DoLink(r,state.get_Value()));
          },
          RouteId:id,
          RouteValue:site
         };
        }
       });
       return{
        $:0,
        $0:{
         $:1,
         $0:site
        },
        $1:t
       };
      },
      DoLink:function(map,va)
      {
       return Route.FromList(map.Ser.call(null,va));
      },
      DoRoute:function(map,route)
      {
       return map.Des.call(null,Route.ToList(route));
      },
      Install:function(key,_arg1)
      {
       var va,site,currentRoute,state,siteTrie,parseRoute,matchValue,glob,site1,updateRoute;
       va=_arg1.$0;
       site=_arg1.$1;
       currentRoute=Routing.InstallMap({
        Des:function(xs)
        {
         return Route.FromList(xs);
        },
        Ser:function(_arg00_)
        {
         return Route.ToList(_arg00_);
        }
       });
       state={
        Bodies:Abbrev.U(),
        CurrentRoute:currentRoute,
        CurrentSite:0,
        Selection:Abbrev.U()
       };
       siteTrie=Trie1.Map(function(prefix)
       {
        return function(_arg11)
        {
         var id;
         id=_arg11.$0;
         return _arg11.$1.call(null,{
          UpdateRoute:function(rest)
          {
           return Routing.OnInternalSiteUpdate(state,id,prefix,rest);
          }
         });
        };
       },site);
       state.Bodies=Routing.ComputeBodies(siteTrie);
       parseRoute=function(route)
       {
        return Trie1.Lookup(siteTrie,Route.ToList(route));
       };
       matchValue=parseRoute(currentRoute.get_Value());
       if(matchValue.$==0)
        {
         site1=matchValue.$0;
         state.CurrentSite=site1.RouteId;
         glob=Var.Create(site1.RouteValue);
        }
       else
        {
         glob=Var.Create(va.$==1?va.$0:Operators.FailWith("Site.Install fails on empty site"));
        }
       state.Selection=glob;
       View1.Sink(function(site2)
       {
        return Routing.OnSelectSite(state,key(site2));
       },glob.get_View());
       updateRoute=function(route)
       {
        var matchValue1;
        matchValue1=parseRoute(route);
        return matchValue1.$==1?null:Routing.OnGlobalRouteChange(state,matchValue1.$0,Route.FromList(matchValue1.$1));
       };
       updateRoute(currentRoute.get_Value());
       View1.Sink(updateRoute,currentRoute.get_View());
       return glob;
      },
      InstallMap:function(rt)
      {
       var cur,_var,onUpdate;
       cur=function()
       {
        return rt.Des.call(null,Route.ToList(Route.ParseHash(window.location.hash)));
       };
       _var=Var.Create(cur(null));
       onUpdate=function()
       {
        var value;
        value=cur(null);
        return!Unchecked.Equals(rt.Ser.call(null,_var.get_Value()),rt.Ser.call(null,value))?_var.set_Value(value):null;
       };
       window.onpopstate=onUpdate;
       window.onhashchange=onUpdate;
       View1.Sink(function(loc)
       {
        var ha;
        ha=Route.MakeHash(Route.FromList(rt.Ser.call(null,loc)));
        return!Route.SameHash(window.location.hash,ha)?void(window.location.hash=ha):null;
       },_var.get_View());
       return _var;
      },
      MergeRouters:function(sites)
      {
       var sites1,merged,value;
       sites1=Seq.toArray(sites);
       merged=Trie1.Merge(Seq.map(function(_arg1)
       {
        return _arg1.$1;
       },sites1));
       value=Seq.tryPick(function(_arg2)
       {
        return _arg2.$0;
       },sites1);
       return merged.$==1?{
        $:0,
        $0:value,
        $1:merged.$0
       }:Operators.FailWith("Invalid Site.Merge: need more prefix disambiguation");
      },
      OnGlobalRouteChange:function(state,site,rest)
      {
       if(state.CurrentSite!==site.RouteId)
        {
         state.CurrentSite=site.RouteId;
         state.Selection.set_Value(site.RouteValue);
        }
       return site.OnRouteChanged.call(null,rest);
      },
      OnInternalSiteUpdate:function(state,ix,prefix,rest)
      {
       return state.CurrentSite===ix?Routing.SetCurrentRoute(state,Route.Append(Route.FromList(prefix),rest)):null;
      },
      OnSelectSite:function(state,_arg1)
      {
       var id;
       id=_arg1.$0;
       if(state.CurrentSite!==id)
        {
         state.CurrentSite=id;
         return state.Bodies.get_Item(id).OnSelect.call(null,null);
        }
       else
        {
         return null;
        }
      },
      SetCurrentRoute:function(state,route)
      {
       return!Unchecked.Equals(state.CurrentRoute.get_Value(),route)?state.CurrentRoute.set_Value(route):null;
      }
     },
     Snap1:{
      Bind:function(f,snap)
      {
       var res,onObs;
       res=Snap1.Create();
       onObs=function()
       {
        return Snap1.MarkObsolete(res);
       };
       Snap1.When(snap,function(x)
       {
        var y;
        y=f(x);
        return Snap1.When(y,function(v)
        {
         return(Snap1.IsForever(y)?Snap1.IsForever(snap):false)?Snap1.MarkForever(res,v):Snap1.MarkReady(res,v);
        },onObs);
       },onObs);
       return res;
      },
      Create:function()
      {
       return Snap1.Make({
        $:3,
        $0:[],
        $1:[]
       });
      },
      CreateForever:function(v)
      {
       return Snap1.Make({
        $:0,
        $0:v
       });
      },
      CreateWithValue:function(v)
      {
       return Snap1.Make({
        $:2,
        $0:v,
        $1:[]
       });
      },
      IsForever:function(snap)
      {
       return snap.State.$==0?true:false;
      },
      IsObsolete:function(snap)
      {
       return snap.State.$==1?true:false;
      },
      Make:function(st)
      {
       return{
        State:st
       };
      },
      Map:function(fn,sn)
      {
       var matchValue,res;
       matchValue=sn.State;
       if(matchValue.$==0)
        {
         return Snap1.CreateForever(fn(matchValue.$0));
        }
       else
        {
         res=Snap1.Create();
         Snap1.When(sn,function(x)
         {
          return Snap1.MarkDone(res,sn,fn(x));
         },function()
         {
          return Snap1.MarkObsolete(res);
         });
         return res;
        }
      },
      Map2:function(fn,sn1,sn2)
      {
       var matchValue,y,y1,res,v1,v2,obs,cont;
       matchValue=[sn1.State,sn2.State];
       if(matchValue[0].$==0)
        {
         if(matchValue[1].$==0)
          {
           y=matchValue[1].$0;
           return Snap1.CreateForever((fn(matchValue[0].$0))(y));
          }
         else
          {
           return Snap1.Map(fn(matchValue[0].$0),sn2);
          }
        }
       else
        {
         if(matchValue[1].$==0)
          {
           y1=matchValue[1].$0;
           return Snap1.Map(function(x)
           {
            return(fn(x))(y1);
           },sn1);
          }
         else
          {
           res=Snap1.Create();
           v1={
            contents:{
             $:0
            }
           };
           v2={
            contents:{
             $:0
            }
           };
           obs=function()
           {
            v1.contents={
             $:0
            };
            v2.contents={
             $:0
            };
            return Snap1.MarkObsolete(res);
           };
           cont=function()
           {
            var matchValue1,x,y2;
            matchValue1=[v1.contents,v2.contents];
            if(matchValue1[0].$==1)
             {
              if(matchValue1[1].$==1)
               {
                x=matchValue1[0].$0;
                y2=matchValue1[1].$0;
                return(Snap1.IsForever(sn1)?Snap1.IsForever(sn2):false)?Snap1.MarkForever(res,(fn(x))(y2)):Snap1.MarkReady(res,(fn(x))(y2));
               }
              else
               {
                return null;
               }
             }
            else
             {
              return null;
             }
           };
           Snap1.When(sn1,function(x)
           {
            v1.contents={
             $:1,
             $0:x
            };
            return cont(null);
           },obs);
           Snap1.When(sn2,function(y2)
           {
            v2.contents={
             $:1,
             $0:y2
            };
            return cont(null);
           },obs);
           return res;
          }
        }
      },
      MapAsync:function(fn,snap)
      {
       var res;
       res=Snap1.Create();
       Snap1.When(snap,function(v)
       {
        return Async.StartTo(fn(v),function(v1)
        {
         return Snap1.MarkDone(res,snap,v1);
        });
       },function()
       {
        return Snap1.MarkObsolete(res);
       });
       return res;
      },
      MarkDone:function(res,sn,v)
      {
       return Snap1.IsForever(sn)?Snap1.MarkForever(res,v):Snap1.MarkReady(res,v);
      },
      MarkForever:function(sn,v)
      {
       var matchValue,q;
       matchValue=sn.State;
       if(matchValue.$==3)
        {
         q=matchValue.$0;
         sn.State={
          $:0,
          $0:v
         };
         return JQueue.Iter(function(k)
         {
          return k(v);
         },q);
        }
       else
        {
         return null;
        }
      },
      MarkObsolete:function(sn)
      {
       var matchValue,ks,ks1;
       matchValue=sn.State;
       if(matchValue.$==1)
        {
         return null;
        }
       else
        {
         if(matchValue.$==2)
          {
           ks=matchValue.$1;
           sn.State={
            $:1
           };
           return JQueue.Iter(function(k)
           {
            return k(null);
           },ks);
          }
         else
          {
           if(matchValue.$==3)
            {
             ks1=matchValue.$1;
             sn.State={
              $:1
             };
             return JQueue.Iter(function(k)
             {
              return k(null);
             },ks1);
            }
           else
            {
             return null;
            }
          }
        }
      },
      MarkReady:function(sn,v)
      {
       var matchValue,q1;
       matchValue=sn.State;
       if(matchValue.$==3)
        {
         q1=matchValue.$0;
         sn.State={
          $:2,
          $0:v,
          $1:matchValue.$1
         };
         return JQueue.Iter(function(k)
         {
          return k(v);
         },q1);
        }
       else
        {
         return null;
        }
      },
      SnapshotOn:function(sn1,sn2)
      {
       var matchValue,res,v,triggered,cont;
       matchValue=[sn1.State,sn2.State];
       if(matchValue[1].$==0)
        {
         return Snap1.CreateForever(matchValue[1].$0);
        }
       else
        {
         res=Snap1.Create();
         v={
          contents:{
           $:0
          }
         };
         triggered={
          contents:false
         };
         cont=function()
         {
          var matchValue1;
          if(triggered.contents)
           {
            matchValue1=v.contents;
            return matchValue1.$==1?Snap1.IsForever(sn2)?Snap1.MarkForever(res,matchValue1.$0):matchValue1.$==1?Snap1.MarkReady(res,matchValue1.$0):null:matchValue1.$==1?Snap1.MarkReady(res,matchValue1.$0):null;
           }
          else
           {
            return null;
           }
         };
         Snap1.When(sn1,function()
         {
          triggered.contents=true;
          return cont(null);
         },function()
         {
          v.contents={
           $:0
          };
          return Snap1.MarkObsolete(res);
         });
         Snap1.When(sn2,function(y)
         {
          v.contents={
           $:1,
           $0:y
          };
          return cont(null);
         },function()
         {
         });
         return res;
        }
      },
      When:function(snap,avail,obsolete)
      {
       var matchValue,v,q2;
       matchValue=snap.State;
       if(matchValue.$==1)
        {
         return obsolete(null);
        }
       else
        {
         if(matchValue.$==2)
          {
           v=matchValue.$0;
           JQueue.Add(obsolete,matchValue.$1);
           return avail(v);
          }
         else
          {
           if(matchValue.$==3)
            {
             q2=matchValue.$1;
             JQueue.Add(avail,matchValue.$0);
             return JQueue.Add(obsolete,q2);
            }
           else
            {
             return avail(matchValue.$0);
            }
          }
        }
      }
     },
     Trans:Runtime.Class({},{
      AnimateChange:function(tr,x,y)
      {
       return(tr.TChange.call(null,x))(y);
      },
      AnimateEnter:function(tr,x)
      {
       return tr.TEnter.call(null,x);
      },
      AnimateExit:function(tr,x)
      {
       return tr.TExit.call(null,x);
      },
      CanAnimateChange:function(tr)
      {
       return(tr.TFlags&1)!==0;
      },
      CanAnimateEnter:function(tr)
      {
       return(tr.TFlags&2)!==0;
      },
      CanAnimateExit:function(tr)
      {
       return(tr.TFlags&4)!==0;
      },
      Change:function(ch,tr)
      {
       return{
        TChange:ch,
        TEnter:tr.TEnter,
        TExit:tr.TExit,
        TFlags:tr.TFlags|1
       };
      },
      Create:function(ch)
      {
       return{
        TChange:ch,
        TEnter:function(t)
        {
         return An.Const(t);
        },
        TExit:function(t)
        {
         return An.Const(t);
        },
        TFlags:1
       };
      },
      Enter:function(f,tr)
      {
       return{
        TChange:tr.TChange,
        TEnter:f,
        TExit:tr.TExit,
        TFlags:tr.TFlags|2
       };
      },
      Exit:function(f,tr)
      {
       return{
        TChange:tr.TChange,
        TEnter:tr.TEnter,
        TExit:f,
        TFlags:tr.TFlags|4
       };
      },
      Trivial:function()
      {
       return{
        TChange:function()
        {
         return function(y)
         {
          return An.Const(y);
         };
        },
        TEnter:function(t)
        {
         return An.Const(t);
        },
        TExit:function(t)
        {
         return An.Const(t);
        },
        TFlags:0
       };
      }
     }),
     Trie1:{
      AllSome:function(xs)
      {
       var e,r,ok,matchValue;
       e=Enumerator.Get(xs);
       r=ResizeArrayProxy.New2();
       ok=true;
       while(ok?e.MoveNext():false)
        {
         matchValue=e.get_Current();
         if(matchValue.$==1)
          {
           r.Add(matchValue.$0);
          }
         else
          {
           ok=false;
          }
        }
       return ok?{
        $:1,
        $0:r.ToArray()
       }:{
        $:0
       };
      },
      Empty:function()
      {
       return{
        $:1
       };
      },
      IsLeaf:function(t)
      {
       return t.$==2?true:false;
      },
      Leaf:function(v)
      {
       return{
        $:2,
        $0:v
       };
      },
      Look:function(key,trie)
      {
       var matchValue,ks,matchValue1;
       matchValue=[trie,key];
       if(matchValue[0].$==2)
        {
         return{
          $:0,
          $0:matchValue[0].$0,
          $1:key
         };
        }
       else
        {
         if(matchValue[0].$==0)
          {
           if(matchValue[1].$==1)
            {
             ks=matchValue[1].$1;
             matchValue1=MapModule.TryFind(matchValue[1].$0,matchValue[0].$0);
             return matchValue1.$==0?{
              $:1
             }:Trie1.Look(ks,matchValue1.$0);
            }
           else
            {
             return{
              $:1
             };
            }
          }
         else
          {
           return{
            $:1
           };
          }
        }
      },
      Lookup:function(trie,key)
      {
       return Trie1.Look(Seq.toList(key),trie);
      },
      Map:function(f,trie)
      {
       return Trie1.MapLoop(Runtime.New(T,{
        $:0
       }),f,trie);
      },
      MapLoop:function(loc,f,trie)
      {
       var x;
       if(trie.$==1)
        {
         return{
          $:1
         };
        }
       else
        {
         if(trie.$==2)
          {
           x=trie.$0;
           return{
            $:2,
            $0:(f(loc))(x)
           };
          }
         else
          {
           return Trie1.TrieBranch(MapModule.Map(function(k)
           {
            return function(v)
            {
             var b;
             b=List.ofArray([k]);
             return Trie1.MapLoop(List.append(loc,b),f,v);
            };
           },trie.$0));
          }
        }
      },
      Mapi:function(f,trie)
      {
       var counter;
       counter={
        contents:0
       };
       return Trie1.Map(function(x)
       {
        var c;
        c=counter.contents;
        counter.contents=c+1;
        return(f(c))(x);
       },trie);
      },
      Merge:function(ts)
      {
       var ts1,matchValue;
       ts1=Seq.toArray(ts);
       matchValue=Arrays.length(ts1);
       return matchValue===0?{
        $:1,
        $0:{
         $:1
        }
       }:matchValue===1?{
        $:1,
        $0:Arrays.get(ts1,0)
       }:Seq.exists(function(t)
       {
        return Trie1.IsLeaf(t);
       },ts1)?{
        $:0
       }:Option.map(function(xs)
       {
        return Trie1.TrieBranch(xs);
       },Trie1.MergeMaps(function(_arg00_)
       {
        return Trie1.Merge(_arg00_);
       },Seq.choose(function(_arg1)
       {
        return _arg1.$==0?{
         $:1,
         $0:_arg1.$0
        }:{
         $:0
        };
       },ts1)));
      },
      MergeMaps:function(merge,maps)
      {
       var x;
       x=Seq.collect(function(table)
       {
        return MapModule.ToSeq(table);
       },maps);
       return Option.map(function(elements)
       {
        return MapModule.OfArray(Seq.toArray(elements));
       },Trie1.AllSome(Seq.map(function(tupledArg)
       {
        var k;
        k=tupledArg[0];
        return Option.map(function(v)
        {
         return[k,v];
        },merge(tupledArg[1]));
       },MapModule.ToSeq(Seq.fold(function(s)
       {
        return function(tupledArg)
        {
         return Trie1.MultiAdd(tupledArg[0],tupledArg[1],s);
        };
       },FSharpMap.New1([]),x)))));
      },
      MultiAdd:function(key,value,map)
      {
       var v;
       v=Runtime.New(T,{
        $:1,
        $0:value,
        $1:Trie1.MultiFind(key,map)
       });
       return map.Add(key,v);
      },
      MultiFind:function(key,map)
      {
       return Operators.DefaultArg(MapModule.TryFind(key,map),Runtime.New(T,{
        $:0
       }));
      },
      Prefix:function(key,trie)
      {
       return Trie1.TrieBranch(FSharpMap.New1(List.ofArray([[key,trie]])));
      },
      ToArray:function(trie)
      {
       var all;
       all=[];
       Trie1.Map(function()
       {
        return function(v)
        {
         return JQueue.Add(v,all);
        };
       },trie);
       return JQueue.ToArray(all);
      },
      TrieBranch:function(xs)
      {
       return xs.get_IsEmpty()?{
        $:1
       }:{
        $:0,
        $0:xs
       };
      }
     },
     UINextPagelet:Runtime.Class({
      Render:function()
      {
       return Doc.RunById(this.divId,this.doc);
      },
      get_Body:function()
      {
       return this.body;
      }
     },{
      New:function(doc)
      {
       var r;
       r=Runtime.New(this,Pagelet.New());
       r.doc=doc;
       r.divId=Fresh.Id();
       r.body=Default.Div(List.ofArray([Default.Id(r.divId)])).get_Body();
       return r;
      }
     }),
     Var:Runtime.Class({
      get_Value:function()
      {
       return Var.Get(this);
      },
      get_View:function()
      {
       return View.FromVar(this);
      },
      set_Value:function(value)
      {
       return Var.Set(this,value);
      }
     },{
      Create:function(v)
      {
       return Runtime.New(Var,{
        Const:false,
        Current:v,
        Snap:Snap1.CreateWithValue(v),
        Id:Fresh.Int()
       });
      },
      Get:function(_var)
      {
       return _var.Current;
      },
      Set:function(_var,value)
      {
       if(_var.Const)
        {
         return null;
        }
       else
        {
         Snap1.MarkObsolete(_var.Snap);
         _var.Current=value;
         _var.Snap=Snap1.CreateWithValue(value);
         return;
        }
      }
     }),
     Var1:Runtime.Class({},{
      GetId:function(_var)
      {
       return _var.Id;
      },
      Observe:function(_var)
      {
       return _var.Snap;
      },
      SetFinal:function(_var,value)
      {
       if(_var.Const)
        {
         return null;
        }
       else
        {
         _var.Const=true;
         _var.Current=value;
         _var.Snap=Snap1.CreateForever(value);
         return;
        }
      },
      Update:function(_var,fn)
      {
       return Var.Set(_var,fn(Var.Get(_var)));
      }
     }),
     View:Runtime.Class({},{
      Convert:function(conv,view)
      {
       return View.ConvertBy(function(x)
       {
        return x;
       },conv,view);
      },
      ConvertBy:function(key,conv,view)
      {
       var state;
       state={
        contents:Dictionary.New12()
       };
       return View.Map(function(xs)
       {
        var prevState,newState,result;
        prevState=state.contents;
        newState=Dictionary.New12();
        result=Arrays.map(function(x)
        {
         var k,res;
         k=key(x);
         res=prevState.ContainsKey(k)?prevState.get_Item(k):conv(x);
         newState.set_Item(k,res);
         return res;
        },Seq.toArray(xs));
        state.contents=newState;
        return result;
       },view);
      },
      ConvertSeqBy:function(key,conv,view)
      {
       var state;
       state={
        contents:Dictionary.New12()
       };
       return View.Map(function(xs)
       {
        var prevState,newState,result;
        prevState=state.contents;
        newState=Dictionary.New12();
        result=Arrays.map(function(x)
        {
         var k,node,n;
         k=key(x);
         if(prevState.ContainsKey(k))
          {
           n=prevState.get_Item(k);
           Var.Set(n.NVar,x);
           node=n;
          }
         else
          {
           node=View.ConvertSeqNode(conv,x);
          }
         newState.set_Item(k,node);
         return node.NValue;
        },Seq.toArray(xs));
        state.contents=newState;
        return result;
       },view);
      },
      ConvertSeqNode:function(conv,value)
      {
       var _var,view;
       _var=Var.Create(value);
       view=View.FromVar(_var);
       return{
        NValue:conv(view),
        NVar:_var,
        NView:view
       };
      },
      CreateLazy:function(observe)
      {
       var cur;
       cur={
        contents:{
         $:0
        }
       };
       return{
        $:0,
        $0:function()
        {
         var matchValue,sn,sn1;
         matchValue=cur.contents;
         if(matchValue.$==1)
          {
           if(!Snap1.IsObsolete(matchValue.$0))
            {
             return matchValue.$0;
            }
           else
            {
             sn=observe(null);
             cur.contents={
              $:1,
              $0:sn
             };
             return sn;
            }
          }
         else
          {
           sn1=observe(null);
           cur.contents={
            $:1,
            $0:sn1
           };
           return sn1;
          }
        }
       };
      },
      CreateLazy2:function(snapFn,_arg3,_arg2)
      {
       var o1,o2;
       o1=_arg3.$0;
       o2=_arg2.$0;
       return View.CreateLazy(function()
       {
        var s1,s2;
        s1=o1(null);
        s2=o2(null);
        return(snapFn(s1))(s2);
       });
      },
      FromVar:function(_var)
      {
       return{
        $:0,
        $0:function()
        {
         return Var1.Observe(_var);
        }
       };
      },
      Map:function(fn,_arg1)
      {
       var observe;
       observe=_arg1.$0;
       return View.CreateLazy(function()
       {
        return Snap1.Map(fn,observe(null));
       });
      },
      Map2:function(fn,v1,v2)
      {
       return View.CreateLazy2(function(_arg10_)
       {
        return function(_arg20_)
        {
         return Snap1.Map2(fn,_arg10_,_arg20_);
        };
       },v1,v2);
      },
      MapAsync:function(fn,_arg4)
      {
       var observe;
       observe=_arg4.$0;
       return View.CreateLazy(function()
       {
        return Snap1.MapAsync(fn,observe(null));
       });
      },
      SnapshotOn:function(def,_arg6,_arg5)
      {
       var o1,o2,res,init;
       o1=_arg6.$0;
       o2=_arg5.$0;
       res=Snap1.CreateWithValue(def);
       init={
        contents:false
       };
       return View.CreateLazy(function()
       {
        var s1,s2;
        s1=o1(null);
        s2=o2(null);
        if(init.contents)
         {
          return Snap1.SnapshotOn(s1,s2);
         }
        else
         {
          Snap1.When(Snap1.SnapshotOn(s1,s2),function()
          {
           return null;
          },function()
          {
           if(!init.contents)
            {
             init.contents=true;
             return Snap1.MarkObsolete(res);
            }
           else
            {
             return null;
            }
          });
          return res;
         }
       });
      },
      UpdateWhile:function(def,v1,v2)
      {
       var value;
       value={
        contents:def
       };
       return View.Map2(function(pred)
       {
        return function(v)
        {
         if(pred)
          {
           value.contents=v;
          }
         return value.contents;
        };
       },v1,v2);
      },
      get_Do:function()
      {
       return Runtime.New(ViewBuilder,{
        $:0
       });
      }
     }),
     View1:Runtime.Class({},{
      Apply:function(fn,view)
      {
       return View.Map2(function(f)
       {
        return function(x)
        {
         return f(x);
        };
       },fn,view);
      },
      Bind:function(fn,view)
      {
       return View1.Join(View.Map(fn,view));
      },
      Const:function(x)
      {
       var o;
       o=Snap1.CreateForever(x);
       return{
        $:0,
        $0:function()
        {
         return o;
        }
       };
      },
      ConvertSeq:function(conv,view)
      {
       return View.ConvertSeqBy(function(x)
       {
        return x;
       },conv,view);
      },
      Join:function(_arg7)
      {
       var observe;
       observe=_arg7.$0;
       return View.CreateLazy(function()
       {
        return Snap1.Bind(function(_arg1)
        {
         return _arg1.$0.call(null,null);
        },observe(null));
       });
      },
      Sink:function(act,_arg8)
      {
       var observe,loop;
       observe=_arg8.$0;
       loop=function()
       {
        return Snap1.When(observe(null),act,function()
        {
         return Async.Schedule(loop);
        });
       };
       return Async.Schedule(loop);
      }
     }),
     ViewBuilder:Runtime.Class({
      Bind:function(x,f)
      {
       return View1.Bind(f,x);
      },
      Return:function(x)
      {
       return View1.Const(x);
      }
     })
    }
   }
  }
 });
 Runtime.OnInit(function()
 {
  Arrays=Runtime.Safe(Global.WebSharper.Arrays);
  Concurrency=Runtime.Safe(Global.WebSharper.Concurrency);
  Array=Runtime.Safe(Global.Array);
  Seq=Runtime.Safe(Global.WebSharper.Seq);
  UI=Runtime.Safe(Global.WebSharper.UI);
  Next=Runtime.Safe(UI.Next);
  Abbrev=Runtime.Safe(Next.Abbrev);
  Fresh=Runtime.Safe(Abbrev.Fresh);
  Collections=Runtime.Safe(Global.WebSharper.Collections);
  HashSet=Runtime.Safe(Collections.HashSet);
  HashSet1=Runtime.Safe(HashSet.HashSet);
  HashSet2=Runtime.Safe(Abbrev.HashSet);
  JQueue=Runtime.Safe(Abbrev.JQueue);
  Slot1=Runtime.Safe(Abbrev.Slot1);
  Unchecked=Runtime.Safe(Global.WebSharper.Unchecked);
  An=Runtime.Safe(Next.An);
  AppendList1=Runtime.Safe(Next.AppendList1);
  Anims=Runtime.Safe(Next.Anims);
  window=Runtime.Safe(Global.window);
  Trans=Runtime.Safe(Next.Trans);
  Option=Runtime.Safe(Global.WebSharper.Option);
  View=Runtime.Safe(Next.View);
  Lazy=Runtime.Safe(Global.WebSharper.Lazy);
  Array1=Runtime.Safe(Abbrev.Array);
  Attrs=Runtime.Safe(Next.Attrs);
  DomUtility=Runtime.Safe(Next.DomUtility);
  Attr=Runtime.Safe(Next.Attr);
  AnimatedAttrNode=Runtime.Safe(Next.AnimatedAttrNode);
  DynamicAttrNode=Runtime.Safe(Next.DynamicAttrNode);
  View1=Runtime.Safe(Next.View1);
  Docs=Runtime.Safe(Next.Docs);
  UINextPagelet=Runtime.Safe(Next.UINextPagelet);
  Doc=Runtime.Safe(Next.Doc);
  List=Runtime.Safe(Global.WebSharper.List);
  Var1=Runtime.Safe(Next.Var1);
  Var=Runtime.Safe(Next.Var);
  Mailbox=Runtime.Safe(Abbrev.Mailbox);
  Operators=Runtime.Safe(Global.WebSharper.Operators);
  NodeSet=Runtime.Safe(Docs.NodeSet);
  DocElemNode=Runtime.Safe(Next.DocElemNode);
  DomNodes=Runtime.Safe(Docs.DomNodes);
  jQuery=Runtime.Safe(Global.jQuery);
  document=Runtime.Safe(Global.document);
  Easing=Runtime.Safe(Next.Easing);
  Easings=Runtime.Safe(Next.Easings);
  FlowBuilder=Runtime.Safe(Next.FlowBuilder);
  Flow1=Runtime.Safe(Next.Flow1);
  Html=Runtime.Safe(Next.Html);
  Elements=Runtime.Safe(Html.Elements);
  T=Runtime.Safe(List.T);
  Input=Runtime.Safe(Next.Input);
  DoubleInterpolation=Runtime.Safe(Next.DoubleInterpolation);
  Key=Runtime.Safe(Next.Key);
  ListModels=Runtime.Safe(Next.ListModels);
  ListModel=Runtime.Safe(Next.ListModel);
  Model=Runtime.Safe(Next.Model);
  Strings=Runtime.Safe(Global.WebSharper.Strings);
  encodeURIComponent=Runtime.Safe(Global.encodeURIComponent);
  Route=Runtime.Safe(Next.Route);
  decodeURIComponent=Runtime.Safe(Global.decodeURIComponent);
  Routing=Runtime.Safe(Next.Routing);
  Router1=Runtime.Safe(Next.Router1);
  Router=Runtime.Safe(Next.Router);
  Trie1=Runtime.Safe(Next.Trie1);
  Dictionary=Runtime.Safe(Collections.Dictionary);
  Snap1=Runtime.Safe(Next.Snap1);
  Async=Runtime.Safe(Abbrev.Async);
  Enumerator=Runtime.Safe(Global.WebSharper.Enumerator);
  ResizeArray=Runtime.Safe(Collections.ResizeArray);
  ResizeArrayProxy=Runtime.Safe(ResizeArray.ResizeArrayProxy);
  MapModule=Runtime.Safe(Collections.MapModule);
  FSharpMap=Runtime.Safe(Collections.FSharpMap);
  Html1=Runtime.Safe(Global.WebSharper.Html);
  Client=Runtime.Safe(Html1.Client);
  Pagelet=Runtime.Safe(Client.Pagelet);
  Default=Runtime.Safe(Client.Default);
  ViewBuilder=Runtime.Safe(Next.ViewBuilder);
  Attributes=Runtime.Safe(Html.Attributes);
  return SvgAttributes=Runtime.Safe(Html.SvgAttributes);
 });
 Runtime.OnLoad(function()
 {
  Runtime.Inherit(UINextPagelet,Pagelet);
  Input.MousePosSt1();
  Input.MouseBtnSt1();
  Input.KeyListenerState();
  Input.ActivateKeyListener();
  Input.ActivateButtonListener();
  SvgAttributes.Z();
  SvgAttributes.YChannelSelector();
  SvgAttributes.Y2();
  SvgAttributes.Y1();
  SvgAttributes.Y();
  SvgAttributes.XChannelSelector();
  SvgAttributes.X2();
  SvgAttributes.X1();
  SvgAttributes.X();
  SvgAttributes.WritingMode();
  SvgAttributes.WordSpacing();
  SvgAttributes.Width();
  SvgAttributes.Visibility();
  SvgAttributes.ViewBox();
  SvgAttributes.Values();
  SvgAttributes.Type();
  SvgAttributes.Transform();
  SvgAttributes.To();
  SvgAttributes.TextRendering();
  SvgAttributes.TextDecoration();
  SvgAttributes.TextAnchor();
  SvgAttributes.TargetY();
  SvgAttributes.TargetX();
  SvgAttributes.SurfaceScale();
  SvgAttributes.Style();
  SvgAttributes.StrokeWidth();
  SvgAttributes.StrokeOpacity();
  SvgAttributes.StrokeMiterLimit();
  SvgAttributes.StrokeLineJoin();
  SvgAttributes.StrokeLineCap();
  SvgAttributes.StrokeDashOffset();
  SvgAttributes.StrokeDashArray();
  SvgAttributes.Stroke();
  SvgAttributes.StopOpacity();
  SvgAttributes.StopColor();
  SvgAttributes.StitchTiles();
  SvgAttributes.StdDeviation();
  SvgAttributes.SpecularExponent();
  SvgAttributes.SpecularConstant();
  SvgAttributes.ShapeRendering();
  SvgAttributes.Seed();
  SvgAttributes.Scale();
  SvgAttributes.Result();
  SvgAttributes.Restart();
  SvgAttributes.RequiredFeatures();
  SvgAttributes.RepeatDur();
  SvgAttributes.RepeatCount();
  SvgAttributes.Radius();
  SvgAttributes.RY();
  SvgAttributes.RX();
  SvgAttributes.R();
  SvgAttributes.PrimitiveUnits();
  SvgAttributes.PreserveAspectRatio();
  SvgAttributes.PreserveAlpha();
  SvgAttributes.PointsAtZ();
  SvgAttributes.PointsAtY();
  SvgAttributes.PointsAtX();
  SvgAttributes.Points();
  SvgAttributes.PointerEvents();
  SvgAttributes.PatternUnits();
  SvgAttributes.PatternTransform();
  SvgAttributes.PatternContentUnits();
  SvgAttributes.PathLength();
  SvgAttributes.PaintOrder();
  SvgAttributes.Overflow();
  SvgAttributes.Order();
  SvgAttributes.Operator();
  SvgAttributes.Opacity();
  SvgAttributes.NumOctaves();
  SvgAttributes.Mode();
  SvgAttributes.Min();
  SvgAttributes.Max();
  SvgAttributes.MaskUnits();
  SvgAttributes.MaskContentUnits();
  SvgAttributes.Mask();
  SvgAttributes.MarkerWidth();
  SvgAttributes.MarkerUnits();
  SvgAttributes.MarkerStart();
  SvgAttributes.MarkerMid();
  SvgAttributes.MarkerHeight();
  SvgAttributes.MarkerEnd();
  SvgAttributes.Local();
  SvgAttributes.LimitingConeAngle();
  SvgAttributes.LightingColor();
  SvgAttributes.LetterSpacing();
  SvgAttributes.KeyTimes();
  SvgAttributes.KeySplines();
  SvgAttributes.Kerning();
  SvgAttributes.KernelUnitLength();
  SvgAttributes.KernelMatrix();
  SvgAttributes.K4();
  SvgAttributes.K3();
  SvgAttributes.K2();
  SvgAttributes.K1();
  SvgAttributes.In2();
  SvgAttributes.ImageRendering();
  SvgAttributes.IN();
  SvgAttributes.Height();
  SvgAttributes.GradientUnits();
  SvgAttributes.GradientTransform();
  SvgAttributes.From();
  SvgAttributes.FontWeight();
  SvgAttributes.FontVariant();
  SvgAttributes.FontStyle();
  SvgAttributes.FontStretch();
  SvgAttributes.FontSizeAdjust();
  SvgAttributes.FontSize();
  SvgAttributes.FontFamily();
  SvgAttributes.FloodOpacity();
  SvgAttributes.FloodColor();
  SvgAttributes.FilterUnits();
  SvgAttributes.FilterRes();
  SvgAttributes.Filter();
  SvgAttributes.FillRule();
  SvgAttributes.FillOpacity();
  SvgAttributes.Fill();
  SvgAttributes.ExternalResourcesRequired();
  SvgAttributes.End();
  SvgAttributes.Elevation();
  SvgAttributes.EdgeMode();
  SvgAttributes.Dur();
  SvgAttributes.DominantBaseline();
  SvgAttributes.Divisor();
  SvgAttributes.Display();
  SvgAttributes.Direction();
  SvgAttributes.DiffuseConstant();
  SvgAttributes.DY();
  SvgAttributes.DX();
  SvgAttributes.D();
  SvgAttributes.Cursor();
  SvgAttributes.ContentStyleType();
  SvgAttributes.ContentScriptType();
  SvgAttributes.ColorRendering();
  SvgAttributes.ColorProfile();
  SvgAttributes.ColorInterpolationFilters();
  SvgAttributes.ColorInterpolation();
  SvgAttributes.Color();
  SvgAttributes.ClipRule();
  SvgAttributes.ClipPathUnits();
  SvgAttributes.ClipPath();
  SvgAttributes.Clip();
  SvgAttributes.Class();
  SvgAttributes.CalcMode();
  SvgAttributes.CY();
  SvgAttributes.CX();
  SvgAttributes.Bias();
  SvgAttributes.Begin();
  SvgAttributes.BaselineShift();
  SvgAttributes.BaseFrequency();
  SvgAttributes.Azimuth();
  SvgAttributes.AttributeType();
  SvgAttributes.AttributeName();
  SvgAttributes.Ascent();
  SvgAttributes.AlignmentBaseline();
  SvgAttributes.Additive();
  SvgAttributes.Accumulate();
  SvgAttributes.AccentHeight();
  Attributes.Wrap();
  Attributes.Width();
  Attributes.Value();
  Attributes.Usemap();
  Attributes.Type();
  Attributes.Title();
  Attributes.Target();
  Attributes.TabIndex();
  Attributes.Summary();
  Attributes.Style();
  Attributes.Step();
  Attributes.Start();
  Attributes.Srcdoc();
  Attributes.SrcLang();
  Attributes.Src();
  Attributes.Spellcheck();
  Attributes.Span();
  Attributes.Sizes();
  Attributes.Size();
  Attributes.Shape();
  Attributes.Selected();
  Attributes.Seamless();
  Attributes.Scoped();
  Attributes.Scope();
  Attributes.Sandbox();
  Attributes.Rows();
  Attributes.RowSpan();
  Attributes.Reversed();
  Attributes.Required();
  Attributes.Rel();
  Attributes.Readonly();
  Attributes.RadioGroup();
  Attributes.PubDate();
  Attributes.Preload();
  Attributes.Poster();
  Attributes.Placeholder();
  Attributes.Ping();
  Attributes.Pattern();
  Attributes.Optimum();
  Attributes.Open();
  Attributes.NoValidate();
  Attributes.Name();
  Attributes.Multiple();
  Attributes.Min();
  Attributes.Method();
  Attributes.Media();
  Attributes.MaxLength();
  Attributes.Max();
  Attributes.Manifest();
  Attributes.Low();
  Attributes.Loop();
  Attributes.List();
  Attributes.Language();
  Attributes.Lang();
  Attributes.Label();
  Attributes.Kind();
  Attributes.KeyType();
  Attributes.ItemProp();
  Attributes.IsMap();
  Attributes.Icon();
  Attributes.ID();
  Attributes.HttpEquiv();
  Attributes.HrefLang();
  Attributes.Href();
  Attributes.High();
  Attributes.Hidden();
  Attributes.Height();
  Attributes.Headers();
  Attributes.FormAction();
  Attributes.Form();
  Attributes.For();
  Attributes.EncType();
  Attributes.Dropzone();
  Attributes.Draggable();
  Attributes.Download();
  Attributes.Disabled();
  Attributes.DirName();
  Attributes.Dir();
  Attributes.Defer();
  Attributes.Default();
  Attributes.Datetime();
  Attributes.Coords();
  Attributes.Controls();
  Attributes.ContextMenu();
  Attributes.ContentEditable();
  Attributes.Content();
  Attributes.Cols();
  Attributes.Color();
  Attributes.ColSpan();
  Attributes.Codebase();
  Attributes.Code();
  Attributes.Class();
  Attributes.Cite();
  Attributes.Checked();
  Attributes.Charset();
  Attributes.Challenge();
  Attributes.Buffered();
  Attributes.Border();
  Attributes.BgColor();
  Attributes.AutoSave();
  Attributes.AutoPlay();
  Attributes.AutoFocus();
  Attributes.AutoComplete();
  Attributes.Async();
  Attributes.Alt();
  Attributes.Align();
  Attributes.Action();
  Attributes.Accesskey();
  Attributes.AcceptCharset();
  Attributes.Accept();
  Easings.CubicInOut();
  DomUtility.Doc();
  Attrs.EmptyAttr();
  Fresh.counter();
  return;
 });
}());
