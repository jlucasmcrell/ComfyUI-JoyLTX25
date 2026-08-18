// Folder filtering for JoyEcho Prompt Source.
// The python side appends a "folder" combo (last, so saved widget order holds);
// this filters the source_file dropdown live on the client. Without this file
// the node still works - the folder choice is validated server-side instead.
import { app } from "../../scripts/app.js";

app.registerExtension({
  name: "joyecho.promptsource.folderfilter",
  nodeCreated(node) {
    if (node.comfyClass !== "JoyEcho_PromptSource") return;
    const fileW = node.widgets?.find((w) => w.name === "source_file");
    const folderW = node.widgets?.find((w) => w.name === "folder");
    if (!fileW || !folderW) return;

    const all = [...(fileW.options?.values ?? [])];
    const folderOf = (v) => {
      const m = /^(TXT|JSON): (.*)$/.exec(v);
      if (!m) return "(all)";
      const rest = m[2];
      const ix = Math.max(rest.lastIndexOf("\\"), rest.lastIndexOf("/"));
      return m[1] + ": " + (ix < 0 ? "(root)" : rest.slice(0, ix));
    };

    const apply = () => {
      const f = folderW.value;
      fileW.options.values =
        !f || f === "(all)" ? [...all] : all.filter((v) => folderOf(v) === f);
      if (!fileW.options.values.includes(fileW.value)) {
        fileW.value = fileW.options.values[0] ?? "";
      }
      node.setDirtyCanvas?.(true, true);
    };

    const prev = folderW.callback;
    folderW.callback = function (...args) {
      prev?.apply(this, args);
      apply();
    };
    // apply once after load so a saved folder value filters immediately
    setTimeout(apply, 0);
  },
});
