// RiftCast: hide the Designer's fallback appearance widgets when a Style node
// is wired.
//
// THE PROBLEM. hair_color / hair_style / wardrobe / distinguishing exist on BOTH
// RiftCast_CharacterDesigner and RiftCast_Style + Wardrobe. When style_block is
// connected the Designer's four are discarded outright (riftcast_generator.py
// builds `appearance` from the style payload and never reads them) - so you set
// hair colour on the Designer, nothing happens, and nothing on screen says why.
// Two identical-looking dropdowns on two nodes, only one of which is live.
//
// WHY NOT JUST DELETE THEM. They are a real fallback: the Designer is used with
// NO Style node in JoyEcho_Multishot_Workflow_PUBLIC.json and in
// RiftCast_Character_Foundry.json. Removing them would break both, and make the
// Style node mandatory for everyone who installed the patch.
//
// WHY NOT JUST A TOOLTIP. A tooltip only helps someone who hovers BEFORE they
// are confused, and the widget labels stay identical either way. The console
// warning in design() fires after the render, which is too late and unseen by
// most users.
//
// THE FIX. Wire a Style node -> the four vanish from the Designer. Unwire it ->
// they come back with their values intact. Nothing is removed from node.widgets,
// so ComfyUI's positional widgets_values serialization is untouched and saved
// workflows keep loading exactly as before.
//
// Everything is wrapped defensively: if a future frontend changes the widget
// API, the worst case is the widgets stay visible (today's behaviour), never a
// broken node.

import { app } from "../../scripts/app.js";

const NODE = "RiftCast_CharacterDesigner";
const INPUT = "style_block";
const SHADOWED = ["hair_color", "hair_style", "wardrobe", "distinguishing"];

function setShown(w, shown) {
    try {
        if (shown) {
            if (w._rcType !== undefined) { w.type = w._rcType; delete w._rcType; }
            if (w._rcSize !== undefined) { w.computeSize = w._rcSize; delete w._rcSize; }
            w.hidden = false;
        } else {
            if (w._rcType === undefined) {
                w._rcType = w.type;
                w._rcSize = w.computeSize;
            }
            // belt and braces: `hidden` is honoured by the current frontend, the
            // zero computeSize collapses the row on older ones.
            w.hidden = true;
            w.computeSize = () => [0, -4];
        }
    } catch (e) {
        console.warn("[RiftCast] widget toggle skipped:", e);
    }
}

function styleWired(node) {
    const i = node.inputs?.find((x) => x?.name === INPUT);
    return !!(i && i.link != null);
}

function apply(node) {
    if (!node?.widgets) return;
    const hide = styleWired(node);
    let changed = false;
    for (const w of node.widgets) {
        if (!SHADOWED.includes(w.name)) continue;
        const already = !!w.hidden;
        if (already !== hide) { setShown(w, !hide); changed = true; }
    }
    if (changed) {
        try {
            const s = node.computeSize();
            node.setSize([Math.max(node.size[0], s[0]), s[1]]);
            node.setDirtyCanvas(true, true);
        } catch (e) { /* sizing is cosmetic - never fail the node over it */ }
    }
}

app.registerExtension({
    name: "riftcast.styleShadow",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE) return;

        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origCreated?.apply(this, arguments);
            apply(this);
            return r;
        };

        const origConn = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const r = origConn?.apply(this, arguments);
            apply(this);
            return r;
        };

        // Loading a saved workflow: links resolve after onConfigure returns, so
        // read the connection state on the next tick or style_block still looks
        // unwired and the widgets flash back in.
        const origConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = origConfigure?.apply(this, arguments);
            setTimeout(() => apply(this), 0);
            return r;
        };
    },
});
