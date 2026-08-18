// JoyEcho widget persistence - values survive node-layout changes.
//
// ComfyUI serializes widgets_values as a POSITIONAL array. Whenever a JoyEcho
// node gains/loses/reorders a widget between saves, every saved value shifts
// into the wrong slot or silently reverts to defaults - the "my settings reset
// after every update" complaint, and the reason the Generate node ships a
// scramble-guard error at all.
//
// Fix: on serialize, ALSO store {widgetName: value} in node.properties (a
// dict - layout-immune). On configure, after the stock positional restore,
// re-apply values BY NAME. Combo values that no longer exist in the current
// option list are skipped so a renamed dropdown option falls back to its
// default instead of erroring.
//
// First save under this extension writes the map; from then on, layout
// changes can never revert your settings again.

import { app } from "../../scripts/app.js";

const PREFIX = "JoyEcho_";
const PROP = "je_widget_values";

app.registerExtension({
    name: "joyecho.widgetPersistence",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!nodeData?.name?.startsWith(PREFIX)) return;

        const origSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (o) {
            origSerialize?.apply(this, arguments);
            if (!this.widgets?.length) return;
            const map = {};
            for (const w of this.widgets) {
                if (w.name !== undefined && w.value !== undefined) {
                    map[w.name] = w.value;
                }
            }
            o.properties = o.properties || {};
            o.properties[PROP] = map;
        };

        const origConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (o) {
            origConfigure?.apply(this, arguments);
            const saved = o?.properties?.[PROP];
            if (!saved || !this.widgets?.length) return;
            for (const w of this.widgets) {
                if (!(w.name in saved)) continue;
                const v = saved[w.name];
                const opts = w.options?.values;
                if (Array.isArray(opts) && !opts.includes(v)) continue; // renamed combo option
                w.value = v;
                w.callback?.(v);
            }
        };
    },
});
