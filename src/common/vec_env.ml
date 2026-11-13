(* Synchronous vector environments. [transitions] always retains final
   observations; [obs] contains reset observations for the next policy call. *)
type ('obs, 'action) t = { envs : ('obs, 'action) Ndenv.t array }
type 'obs step = { transitions : 'obs Ndenv.step array; obs : 'obs array }

let create envs =
  if Array.length envs = 0 then invalid_arg "Vec_env.create: empty";
  Array.iter (fun env ->
    if env.Ndenv.observation_space <> envs.(0).Ndenv.observation_space
       || env.action_space <> envs.(0).action_space
    then invalid_arg "Vec_env.create: incompatible spaces") envs;
  { envs = Array.copy envs }

let length t = Array.length t.envs
let reset t = Array.map (fun env -> env.Ndenv.reset ()) t.envs
let step t actions =
  if Array.length actions <> length t then invalid_arg "Vec_env.step: action count";
  let transitions = Array.map2 (fun env a -> env.Ndenv.step a) t.envs actions in
  let obs = Array.mapi (fun i st ->
    if st.Ndenv.terminated || st.truncated then t.envs.(i).reset () else st.obs) transitions in
  { transitions; obs }
