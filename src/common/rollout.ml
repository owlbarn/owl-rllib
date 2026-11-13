(* Flat, time-major layout: index = time * n_envs + environment.
   next_values refer to the transition's final observation, never an auto-reset.
   Truncation bootstraps delta but stops the GAE recursion across episodes. *)
let gae ~n_envs ~gamma ~lambda ~rewards ~values ~next_values ~terminated ~truncated =
  let n = Array.length rewards in
  if n_envs <= 0 || n mod n_envs <> 0
     || List.exists (fun k -> k <> n)
       [ Array.length values; Array.length next_values; Array.length terminated; Array.length truncated ]
  then invalid_arg "Rollout.gae: inconsistent shapes";
  let carry = Array.make n_envs 0. in
  let advantages = Array.make n 0. in
  for i = n - 1 downto 0 do
    let env = i mod n_envs in
    let delta = Buffer.td_target ~gamma ~reward:rewards.(i) ~terminated:terminated.(i)
        ~next_value:next_values.(i) -. values.(i) in
    let tail = if terminated.(i) || truncated.(i) then 0. else carry.(env) in
    carry.(env) <- delta +. gamma *. lambda *. tail;
    advantages.(i) <- carry.(env)
  done;
  advantages, Array.map2 ( +. ) advantages values

let permutation ~rng n =
  let indices = Array.init n Fun.id in
  for i = n - 1 downto 1 do
    let j = Random.State.int rng (i + 1) in
    let old = indices.(i) in indices.(i) <- indices.(j); indices.(j) <- old
  done;
  indices

module Make (A : Owl_types_ndarray_algodiff.Sig) = struct
  type transition =
    { obs : A.arr; action : A.arr; log_prob : float; value : float
    ; reward : float; next_value : float; terminated : bool; truncated : bool }
  type t = { data : transition array; advantages : float array; returns : float array }
  let create ~n_envs ~gamma ~lambda data =
    let advantages, returns = gae ~n_envs ~gamma ~lambda
        ~rewards:(Array.map (fun x -> x.reward) data)
        ~values:(Array.map (fun x -> x.value) data)
        ~next_values:(Array.map (fun x -> x.next_value) data)
        ~terminated:(Array.map (fun x -> x.terminated) data)
        ~truncated:(Array.map (fun x -> x.truncated) data) in
    { data; advantages; returns }
end
module S = Make (Owl.Neural.S.Algodiff.A)
module D = Make (Owl.Neural.D.Algodiff.A)
