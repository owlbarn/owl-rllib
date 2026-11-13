(* Shared tensor boundaries; algorithm update loops stay in their own files. *)
module Make (Graph : Owl_neural_graph_sig.Sig) = struct
  include Owl_neural_generic.Flatten (Graph)
  open Algodiff
  module E = Ndenv.Make (A)
  module B = Buffer.Make (A)
  module Distribution = Dist.Make (Algodiff)

  let scalar x = F (A.float_to_elt x)
  let column xs = Arr (A.of_array (Array.map A.float_to_elt xs) [| Array.length xs; 1 |])
  let batch xs = Arr (B.batch xs)
  let one_hot n actions =
    Arr (A.of_arrays (Array.map (fun a -> Array.init n (fun j -> A.float_to_elt (if a = j then 1. else 0.))) actions))
  let values x =
    let flat = A.reshape x [| A.numel x |] in
    Array.init (A.numel x) (fun i -> A.get flat [| i |] |> A.elt_to_float)
  let predict nn obs = Graph.run (batch [| obs |]) nn |> unpack_arr |> values
  let n_actions (env : int E.t) = match env.action_space with
    | Space.Discrete n -> n
    | _ -> invalid_arg "algorithm requires discrete actions"
  let adam lr = Params.config ~learning_rate:(Learning_Rate.Adam (lr, 0.9, 0.999)) 1.
  let update nn stepper loss =
    let ws, gs = Graph.backward nn loss in
    Graph.update nn (Stepper.step stepper ws gs);
    Gc.minor ()
  let mlp ~seed ~shape ~hidden ~outputs =
    Owl_stats_prng.init seed;
    let input = Graph.input shape |> Graph.flatten in
    let body = List.fold_left (fun x n -> Graph.linear n ~act_typ:Activation.Relu x) input hidden in
    let nn = Graph.linear outputs body |> Graph.get_network in
    Graph.init nn;
    nn

  (* Orthogonal weights times [gain], like PyTorch's orthogonal_: Gram-Schmidt
     on a Gaussian matrix drawn from Owl's generator, as the default init is. *)
  let orthogonal gain = Init.Custom (fun shape ->
    let rows = shape.(0) and cols = shape.(1) in
    let n = max rows cols and k = min rows cols in
    let g = values (A.gaussian [| k; n |]) in
    let q = Array.init k (fun i -> Array.sub g (i * n) n) in
    let dot u v = Array.fold_left ( +. ) 0. (Array.map2 ( *. ) u v) in
    Array.iteri (fun i v ->
      for j = 0 to i - 1 do
        let d = dot v q.(j) in
        Array.iteri (fun t x -> v.(t) <- x -. d *. q.(j).(t)) v
      done;
      let norm = sqrt (dot v v) in
      Array.iteri (fun t x -> v.(t) <- x /. norm) v) q;
    (* k orthonormal vectors of length n: the columns of a tall matrix, the rows of a wide one *)
    Arr (A.init [| rows; cols |] (fun i ->
      let r = i / cols and c = i mod cols in
      A.float_to_elt (gain *. if rows >= cols then q.(c).(r) else q.(r).(c)))))

  (* CleanRL's PPO networks use tanh and orthogonal initialisation: gain
     sqrt 2 for hidden layers and a chosen gain for the output layer. *)
  type init = Default | Orthogonal of float (* output-layer gain *)

  let hidden_layers ~activation ~init ~shape hidden =
    let init_typ = match init with Default -> None | Orthogonal _ -> Some (orthogonal (sqrt 2.)) in
    List.fold_left (fun x n -> Graph.linear ?init_typ n ~act_typ:activation x) (Graph.input shape |> Graph.flatten) hidden

  let output_layer ~init n x =
    Graph.linear ?init_typ:(match init with Default -> None | Orthogonal gain -> Some (orthogonal gain)) n x

  let build ~seed node =
    Owl_stats_prng.init seed;
    let nn = Graph.get_network (node ()) in
    Graph.init nn;
    nn

  let mlp_with ~activation ~init ~seed ~shape ~hidden ~outputs =
    build ~seed (fun () -> hidden_layers ~activation ~init ~shape hidden |> output_layer ~init outputs)

  (* Output [mean; log_std] for Policy.gaussian, with a state-independent
     log_std: a bias-only layer (its input is zeroed), so log_std is one
     learned vector, initially 0. *)
  let gaussian_mlp ~activation ~init ~seed ~shape ~hidden ~actions =
    build ~seed (fun () ->
      let features = hidden_layers ~activation ~init ~shape hidden in
      let log_std = Graph.lambda (fun x -> Maths.(x * scalar 0.)) features |> Graph.linear actions in
      Graph.concatenate 1 [| output_layer ~init actions features; log_std |])
  (* NHWC images [height; width; channels]. Each (kernel, stride, channels)
     is a VALID ReLU convolution; the hidden layers follow the flattened map. *)
  let cnn ~seed ~shape ~convs ~hidden ~outputs =
    Owl_stats_prng.init seed;
    let features, _ = List.fold_left (fun (x, input) (kernel, stride, output) ->
      Graph.conv2d ~padding:Owl_types.VALID [| kernel; kernel; input; output |] [| stride; stride |]
        ~act_typ:Activation.Relu x, output) (Graph.input shape, shape.(2)) convs in
    let body = List.fold_left (fun x n -> Graph.linear n ~act_typ:Activation.Relu x) (Graph.flatten features) hidden in
    let nn = Graph.linear outputs body |> Graph.get_network in
    Graph.init nn;
    nn
end
