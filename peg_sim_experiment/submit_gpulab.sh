#!/bin/bash
# submit_gpulab.sh <run>   e.g. peg_200, peg_c_200 -- uploads the run's dataset + config to /project_ghent/peg_sim and
# submits the GPULab training job. Needs a running upload gateway (the gpulab skill's bottle/gateway.sh):
#   eval "$(bash ~/.claude/skills/gpulab/bottle/gateway.sh)"
# Reuses /project_ghent/bottle/start_bottle_training.sh (same image, same fork install, same HF/wandb login).
set -eo pipefail
[ -n "$GW_HOST" ] && [ -n "$GW_PORT" ] || { echo "GW_HOST/GW_PORT not set -- start a gateway first"; exit 1; }
RUN="$1"; HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(dirname "$HERE"); G=$HERE/generated; mkdir -p "$G"
P="$HOME/Downloads/login_ilabt_imec_be_rtalwar@ugent.be.pem"; export GPULAB_CERT="$P"
R() { ssh -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -i "$P" -p $GW_PORT -o ProxyCommand="ssh -o BatchMode=yes -o IdentitiesOnly=yes -i '$P' fffrtalwaru@bastion.ilabt.imec.be -W %h:%p" root@$GW_HOST "$@" 2>&1 | { grep -v -E 'ksshaskpass|QDBus' || true; }; }
CFG=$HERE/configs/$RUN.json; [ -f "$CFG" ] || { echo "no config $CFG"; exit 1; }
DS=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['dataset']['repo_id'])" "$CFG")
python3 - "$CFG" "$G/$RUN.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c["dataset"]["root"] = "/project_ghent/peg_sim/" + c["dataset"]["root"].removeprefix("datasets/peg_sim/")
c["output_dir"] = "/project_ghent/peg_sim/" + c["output_dir"].removeprefix("outputs/train/peg_sim/")
json.dump(c, open(sys.argv[2], "w"), indent=2)
print("config:", c["dataset"]["root"], "->", c["output_dir"], "| steps", c["steps"], "| action", c["policy"]["output_features"]["action"]["shape"])
PY
R "mkdir -p /project_ghent/peg_sim/configs && chmod a+rwX /project_ghent/peg_sim /project_ghent/peg_sim/configs"
# Never overwrite a dataset in place: a running job reading it dies with "Stale file handle" (bottle, 2026-10-15).
if [ "$(R "test -d /project_ghent/peg_sim/$DS && echo EXISTS")" = EXISTS ] && [ -z "$FORCE_UPLOAD" ]; then
  echo "dataset $DS already on /project_ghent, not re-uploading"
else
  (cd $REPO/datasets/peg_sim && tar cf - $DS) | R "tar xf - -C /project_ghent/peg_sim && chmod -R a+rwX /project_ghent/peg_sim/$DS"
  echo "uploaded $DS: $(R "du -sh /project_ghent/peg_sim/$DS | cut -f1")"
fi
tar czf - -C $G $RUN.json | R 'tar xzf - -C /project_ghent/peg_sim/configs && chmod a+rw /project_ghent/peg_sim/configs/*'
cat > $G/job_$RUN.json <<JSON
{
  "name": "$RUN",
  "request": {
    "docker": {
      "image": "ramantalwarnr2/mt-lerobot:py312-cu128",
      "command": "bash /project_ghent/bottle/start_bottle_training.sh /project_ghent/peg_sim/configs/$RUN.json",
      "environment": {}, "portMappings": [],
      "storage": [{"hostPath": "/project_ghent", "containerPath": "/project_ghent"}]
    },
    "resources": {"gpus": 1, "gpuModel": ["H200 NVL", "5090", "PRO 6000"], "cpus": 8, "cpuMemoryGb": 32},
    "scheduling": {"maxDuration": "2 days", "minDuration": "1 day"},
    "extra": {"emailOnEnd": ["raman.talwar@ugent.be"]}
  }
}
JSON
~/.local/bin/gpulab-cli submit --project phd-raman-talwar < $G/job_$RUN.json
