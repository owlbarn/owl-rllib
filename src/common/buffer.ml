(* Float64 return accumulation; the final value supports time-limit bootstrap. *)
let returns ~gamma ?(last_value = 0.) rewards =
  let g = ref last_value in
  let out = Array.make (Array.length rewards) 0. in
  for i = Array.length rewards - 1 downto 0 do
    g := rewards.(i) +. gamma *. !g;
    out.(i) <- !g
  done;
  out

let td_target ~gamma ~reward ~terminated ~next_value =
  reward +. if terminated then 0. else gamma *. next_value

module Make (A : Owl_types_ndarray_algodiff.Sig) = struct
  type transition =
    { obs : A.arr; action : int; reward : float; next_obs : A.arr
    ; terminated : bool; truncated : bool }

  type t = { data : transition option array; mutable size : int; mutable next : int }
  let create capacity =
    if capacity <= 0 then invalid_arg "Buffer.create: positive capacity required";
    { data = Array.make capacity None; size = 0; next = 0 }
  let length t = t.size
  let add t tr =
    t.data.(t.next) <- Some { tr with obs = A.copy tr.obs; next_obs = A.copy tr.next_obs };
    t.next <- (t.next + 1) mod Array.length t.data;
    t.size <- min (t.size + 1) (Array.length t.data)
  (* Uniform sampling with replacement. Returned observations are owned copies. *)
  let sample ~rng ~batch_size t =
    if t.size = 0 || batch_size <= 0 then invalid_arg "Buffer.sample: empty batch or buffer";
    Array.init batch_size (fun _ ->
      let tr = Option.get t.data.(Random.State.int rng t.size) in
      { tr with obs = A.copy tr.obs; next_obs = A.copy tr.next_obs })
  let batch observations =
    A.stack ~axis:0 observations
end
module S = Make (Owl.Neural.S.Algodiff.A)
module D = Make (Owl.Neural.D.Algodiff.A)
