// Learn more about F# at http://docs.microsoft.com/dotnet/fsharp

open System
open FSharp.Plotly

let f181 (p:seq<double>) = 
    let x = Seq.item 0 p
    let y = Seq.item 1 p

    let a = Math.Exp(8.0*x - 13.0*y - 21.0)
    let b = Math.Exp(21.0*y - 13.0*x - 34.0)
    let c = 0.0001*Math.Exp(x+y)

    a + b + c

let g181 (p:seq<double>) =
    let x = Seq.item 0 p
    let y = Seq.item 1 p
    let a = Math.Exp(8.0*x-13.0*y+21.)
    let b = Math.Exp(21.0*y-13.0*x-34.)
    let c = 0.0001*Math.Exp(x+y)

    [8.0*a-13.0*b+c; -13.0*a+21.0*b+c]

let f182 (p:seq<double>) =
    let x = Seq.item 0 p
    let y = Seq.item 1 p
    let a = 3.*x*y - 2.*y
    let b = (x**2. + y**2. - 1.1)
    let c = 1000.*b*Math.Exp(10.*b)

    a + b + c

let g182 (p:seq<double>) =
    let x = Seq.item 0 p
    let y = Seq.item 1 p
    let a = 3.*x*y - 2.*y
    let b = x**2.+y**2.-1.1
    let c = Math.Exp(10.*b)

    [3.*y + 1000.*(2.*x*c + 20.*x*b*c); 3.*x-2. + 1000.*(2.*y*c + 20.*y*b*c)]

let f183 (p:seq<double>) = 
    let tx = Seq.append (Seq.singleton -1.) p
    let x = Seq.append tx (Seq.singleton 1.)
    let xl = List.ofSeq x
    let rec f (xs) =
        match xs with
        | x1::x2::rst -> 0.5*((x2 - x1)**2.) + 0.0625*((1. - x1**2.)**2.) + (f (x2::rst))
        | _ -> 0.
    f xl

let g183 (p:seq<double>) = 
    let tx = Seq.append (Seq.singleton -1.) p
    let x = Seq.append tx (Seq.singleton 1.)
    let xl = List.ofSeq x

    let rec g (xs) =
        match xs with
        | a::b::c::rest -> 2.*b - c - a - 0.25*(b - b**3.)::g(b::c::rest)
        | _ -> []

    g xl |> Seq.ofList

// Gradient Descent w/ Backtracking
let gradientDescent f gradF x0 f0 dt0=
    let rec descentLoop x (fval:double) dt = 
        let newf = (f x)
        let newg = (gradF x)
        if (Math.Abs (newf - fval)) < 1e-10 then
            printfn "Delta: %A" (Math.Abs (newf - fval))
            x
        else
            let newx = Seq.zip x newg |> Seq.map (fun (x, dx) -> x - dt*dx)
            let inter = (f newx)
            if (inter > fval) then
                descentLoop x (fval + 1.0) (0.5*dt)
            else 
                let newt = 1.1*dt
                descentLoop newx newf newt

    descentLoop x0 f0 dt0

[<EntryPoint>]
let main argv =
    printfn "Question 18.1: Optimization"
    let xstar = gradientDescent f181 g181 [0.;0.] 6000. 1e-1
    printfn "Optimal point: %A" xstar
    printfn "Optimal Function Value: %A" (f181 xstar)

    printfn "Question 18.2: Optimization"
    let xstar = gradientDescent f182 g182 [0.;0.] 6000. 1e-3
    printfn "Optimal point: %A" xstar
    printfn "Optimal Function Value: %A" (f182 xstar)

    let x0 = (Seq.init 99 (fun x -> 0.1))
    let f0 = (f183 x0) + 100.

    printfn "Third problem"

    printfn "Initial Gradient: %A" (g183 x0)
    let xstar = gradientDescent f183 g183 x0 f0 1e-1
    printfn "Optimal Point: %A" xstar
    printfn "Optimal Function Value: %A" (f183 xstar)
    let indices = (Seq.init 99 id)
    let chart183 = Chart.Line(indices, xstar,
                        Name="Optimal Point for 18.3",
                        ShowMarkers=true,
                        MarkerSymbol=StyleParam.Symbol.Square)
                    |> Chart.withLineStyle(Width=2,Dash=StyleParam.DrawingStyle.Dot)
    chart183 |> Chart.Show
    0 // return an integer exit code