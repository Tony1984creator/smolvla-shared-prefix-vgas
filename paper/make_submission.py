"""Build submission figures and tables directly from the archived result ledgers."""
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

from figstyle import COLORS as C, LABELS, MARKERS, WIDTH, grid, panel, plt, save
from width_time_figure import draw as width_time

PAPER = Path(__file__).resolve().parent
RESULTS = PAPER.parent / "results"
ASSETS = PAPER / "assets"
COHORTS = (
    ("Object calibration", "object-sweep", "B-BC", "B-Bo4", "front5", "late"),
    ("Object new layouts", "object-fresh", "single", "dense", "front", "late"),
    ("Long Task2", "long-task2", "Z", "SharedBo4", "front8", None),
    ("Goal", "goal-window", "single", "dense", "front", "late"),
    ("Spatial", "spatial-window", "single", "dense", "front", "late"),
    ("Plus lighting", "plus-light-window", "single", "dense", "front", None),
)
ARMS = tuple(LABELS)
SLASH = chr(92)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load(name):
    return read(RESULTS / f"{name}.json")


def rows(name):
    with (RESULTS / "scheduled_selection" / f"{name}.csv").open(newline="") as stream:
        return list(csv.DictReader(stream))


def cohort_data():
    summary = load("scheduled_selection/summary")
    for name, key, *arms in COHORTS:
        ledger = rows(key)
        if key in ("object-sweep", "object-fresh"):
            ledger += [row for row in rows("late-calibration" if key == "object-sweep" else "late-object")
                       if row["arm"] == "late"]
        stats = load("scheduled_selection/goal-window") if key == "goal-window" else summary[key]
        source = stats["totals"]
        selected = {}
        for tag, arm in zip(ARMS, arms, strict=True):
            if arm is None:
                continue
            subset = [row for row in ledger if row["arm"] == arm]
            assert subset
            count = sum(int(row["success"]) for row in subset)
            assert count == source[arm].get("wins", source[arm].get("success"))
            selected[tag] = (count, len(subset), subset)
        contrast = stats["paired"]["front_vs_dense"] if key == "goal-window" else stats["paired"][arms[2]]["dense"]
        saved = 100 * (1 - source[arms[2]]["calls"] / source[arms[1]]["calls"])
        yield name, key, selected, contrast, saved


def error(ax, x, y, limits, color=C["ink"], **kwargs):
    bounds = np.asarray(limits).reshape(-1, 2).T
    ax.errorbar(x, y, yerr=np.abs(bounds - np.atleast_1d(y)),
                color=color, capsize=2.4, elinewidth=0.8, **kwargs)


def example():
    config = read(ASSETS / "object-example.json")
    return config["query_widths"]


def teaser():
    graph = load("graph-closed-loop/summary")["arms"]
    widths = example()
    fig = plt.figure(figsize=(WIDTH, 1.65))
    scene = fig.add_axes((0.015, 0.11, 0.60, 0.81))
    scene.set_axis_off()
    query_centers = np.linspace(0.107, 0.867, len(widths))
    for q in (1, 3, 5, 7, 9):
        center = query_centers[q - 1]
        view = scene.inset_axes((center - 0.0855, 0.31, 0.171, 0.62))
        view.imshow(plt.imread(ASSETS / f"object-q{q:02}.png"), interpolation="none")
        view.set_xticks([])
        view.set_yticks([])
        for spine in view.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.1)
            spine.set_edgecolor(C["front" if widths[q-1] == 4 else "single"])
        scene.text(center, 0.26, f"q{q}", ha="center", va="top", transform=scene.transAxes)
    for center, width in zip(query_centers, widths, strict=True):
        scene.add_patch(Rectangle((center - 0.041, 0.073), 0.082, 0.081,
                                  transform=scene.transAxes, facecolor=C["front" if width == 4 else "single"]))
    for width, caption, color in ((4, "Four proposals", C["front"]), (1, "One proposal", C["single"])):
        center = np.mean(query_centers[np.asarray(widths) == width])
        scene.text(center, 0, caption, transform=scene.transAxes, color=color, ha="center", va="center")
    ax = fig.add_axes((0.73, 0.22, 0.25, 0.64))
    for key, tag, offset in (("single", "single", (0, 6)), ("dense", "dense", (-2, 7)), ("front5", "front", (2, 7))):
        arm = graph[key]
        x = arm["mean_episode_ms"]
        y = 100 * arm["successes"] / arm["episodes"]
        ax.scatter(x, y, color=C[tag], marker=MARKERS[tag], s=19, zorder=3)
        ax.annotate(LABELS[tag], (x, y), xytext=offset, textcoords="offset points", ha="center",
                    fontsize=8, color=C[tag])
    ax.annotate("", (graph["front5"]["mean_episode_ms"] + 5, 55),
                (graph["dense"]["mean_episode_ms"] - 5, 55),
                arrowprops=dict(arrowstyle="->", color=C["front"], lw=.85))
    ax.set(xlim=(805, 971), ylim=(46, 60), xlabel="Core ms / episode",
           ylabel="Success (%)", xticks=(825, 875, 925), yticks=(48, 52, 56))
    grid(ax)
    ax.tick_params(axis="y", labelsize=8)
    panel(scene, "a", x=.018, y=.946, transform=fig.transFigure, va="top")
    panel(ax, "b", x=.673, y=.946, transform=fig.transFigure, va="top")
    save(fig, "teaser")


def method():
    fig, ax = plt.subplots(figsize=(WIDTH, 2.12))
    fig.subplots_adjust(left=.018, right=.988, top=.955, bottom=.08)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    panel(ax, "a", x=0, y=.955)
    for i, (filename, label) in enumerate((("object-q01.png", "External"), ("object-q01-wrist.png", "Wrist"))):
        im = ax.inset_axes((.022 + .107*i, .58, .102, .34))
        im.imshow(plt.imread(ASSETS / filename), interpolation="none")
        im.set_axis_off()
        ax.text(.073+.107*i, .557, label, ha="center", va="center", fontsize=7)
    ax.text(.124, .48, "Images, text, robot state", ha="center", fontsize=7)

    def box(x, y, w, h, title, sub, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.012,rounding_size=.022",
                                   facecolor=color, edgecolor=C["single"], linewidth=.65))
        ax.text(x+w/2, y+.65*h, title, ha="center", va="center", weight="bold", fontsize=8)
        ax.text(x+w/2, y+.29*h, sub, ha="center", va="center", fontsize=7)

    def arrow(x0, y0, x1, y1, rad=0):
        ax.annotate("", (x1, y1), (x0, y0),
                    arrowprops=dict(arrowstyle="->", lw=.9, color=C["ink"],
                                    connectionstyle=f"arc3,rad={rad}"))

    box(.282,.66,.14,.194,"Encode once","Prefix KV",C["pale_blue"])
    box(.493,.66,.147,.194,"Action expert","4 noise draws",C["pale_orange"])
    box(.710,.66,.125,.194,"VGAS critic","Rank chunks",C["pale_blue"])
    box(.898,.66,.089,.194,"Execute","≤20 steps",C["light"])
    for x0,x1 in ((.233,.274),(.432,.483),(.653,.701),(.847,.888)):
        arrow(x0,.756,x1,.756)
    arrow(.363,.657,.761,.656,rad=.30)
    ax.text(.56,.416,"Shared observation features",ha="center",fontsize=7)
    panel(ax, "b", x=0, y=.35)
    for i in range(14):
        x = .03+.0675*i
        color = C["front" if i < 5 else "single"]
        ax.add_patch(Rectangle((x,.13), .060,.115, color=color))
        ax.text(x+.03,.187,str(i+1),ha="center",va="center",fontsize=7,color="white")
    ax.text(.03,.067,"Bo4 at queries 1–5",ha="left",color=C["front"])
    ax.text(.368,.067,"Single proposal at subsequent queries",ha="left",color=C["single"])
    save(fig, "method-overview")


def marginal():
    values = load("scheduled_selection/marginal-value")["segments"]
    fig, (a,b) = plt.subplots(1,2,figsize=(WIDTH,2.0),layout="constrained")
    x = np.arange(len(values))
    delta = [s["delta_pp"] for s in values]
    a.bar(x,delta,width=.55,color=[C["front"] if i <= 1 else C["dense"] for i in x])
    error(a, x, delta, [[s["ci95_low_pp"],s["ci95_high_pp"]] for s in values],fmt="none")
    a.axhline(0,color=C["single"],lw=.6)
    a.set(xlabel="Query block",ylabel="Increment (pp)",xticks=x,
          xticklabels=[f'{s["first_query"]}–{s["last_query"]}' for s in values],ylim=(-3.1,12.8))
    k=[0]+[s["cutoff"] for s in values]
    b.plot(k,[0]+[s["capture_pct"] for s in values],marker="o",ms=3.5,label="Gain captured",color=C["front"])
    b.plot(k,[0]+[s["full_call_share_pct"] for s in values],marker="s",ms=3.5,ls="--",label="Calls used",color=C["dense"])
    late = load("scheduled_selection/summary")["object-sweep"]["paired"]["late"]["single"]["pp"]
    full = values[-1]["cumulative_gain_pp"]
    b.scatter([9],[100*late/full],s=18,marker=MARKERS["late"],color=C["late"],label="Late (9 slots)")
    b.axvline(k[2],color=C["grid"],lw=.8)
    b.annotate(f'{values[1]["capture_pct"]:.0f}% gain / {values[1]["full_call_share_pct"]:.0f}% calls',
               (k[2],values[1]["capture_pct"]), xytext=(-18,13),textcoords="offset points",
               fontsize=7,color=C["front"],ha="left")
    b.set(xlabel="Planned four-way slots",ylabel="Share of full Bo4 (%)",xticks=k,ylim=(-22,127))
    b.legend(loc="lower right",fontsize=7)
    for ax, label in ((a,"a"),(b,"b")):
        panel(ax,label)
        grid(ax)
    save(fig,"marginal-value")


def bootstrap_success(ledger):
    per_task = defaultdict(lambda: defaultdict(list))
    for row in ledger:
        per_task[row["task"]][row.get("layout") or row["case"]].append(int(row["success"]))
    rng = np.random.default_rng(20261128)
    totals = np.zeros((10000,2))
    for layouts in per_task.values():
        weights = np.array([(sum(x),len(x)) for x in layouts.values()])
        draws = rng.integers(len(weights),size=(len(totals),len(weights)))
        totals += weights[draws].sum(1)
    return (100 * np.percentile(totals[:,0]/totals[:,1],[2.5,97.5])).tolist()


def cohorts(data):
    fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
    fig.subplots_adjust(top=.81,bottom=.27,left=.097,right=.985)
    names = ("Object\ncalibration","Object\nnew","Long\nTask2","Goal","Spatial","Plus\nlighting")
    for i, (_,_,selected,_,saved) in enumerate(data):
        for j, tag in enumerate(ARMS):
            if tag not in selected:
                continue
            count, count_all, ledger = selected[tag]
            rate = 100 * count/count_all
            pos = i + (j-1.5)*.18
            ax.bar(pos,rate,width=.15,color=C[tag],label=LABELS[tag] if i==0 else "_nolegend_")
            error(ax,pos,rate,bootstrap_success(ledger),fmt="none")
        ax.text(i,-.265,f"−{saved:.1f}%",transform=ax.get_xaxis_transform(),color=C["front"],ha="center",fontsize=8)
    ax.set(ylabel="Success (%)",xticks=np.arange(6),xticklabels=names,ylim=(0,72),xlim=(-.56,5.56))
    grid(ax)
    ax.legend(loc="lower center",bbox_to_anchor=(.5,1.015),ncol=4)
    save(fig,"cross-suite")


def mechanism():
    d = load("scheduled_selection/critic-oracle-decomposition")
    q = list(d["by_query"].values())
    x=np.arange(1,4)
    fig, axes = plt.subplots(1,3,figsize=(WIDTH,1.74),layout="constrained")
    left, mid, right = axes
    left.plot(x,[100*s["mixed_rate"] for s in q],marker="o",ms=4,color=C["dense"],label="Mixed pools")
    left.plot(x,[s["conditional_oracle_gap_pp"] for s in q],marker="s",ms=4,
              color=C["oracle"],ls="--",label="Uniform failure")
    left.set(ylabel="Rate (%)",ylim=(25,60))
    left.legend(loc="upper center",fontsize=7)
    mid.bar(x,[100*s["capture"] for s in q],width=.51,color=C["green"])
    mid.set(ylabel="Gap captured (%)",ylim=(0,23))
    y=[s["critic_gain_pp"] for s in q]
    right.bar(x,y,width=.51,color=C["front"])
    error(right,x,y,[s["critic_gain_ci95_pp"] for s in q],fmt="none")
    right.axhline(0,color=C["single"],lw=.55)
    right.set(ylabel="Critic − random (pp)",ylim=(-2,9))
    for a,l in zip(axes,"abc"):
        a.set(xlabel="Reference query",xticks=x)
        grid(a)
        panel(a,l)
    save(fig,"mechanism")


def system_fig():
    fig, axes = plt.subplots(1,3,figsize=(WIDTH,1.86),layout="constrained",
                             gridspec_kw={"width_ratios":[1,1.13,.86]})
    a,b,c=axes
    obj=load("object-paired")
    vals=[1000*mean(t for r in obj[k] for t in r["decision_seconds"]) for k in ("B-OrigBo4","B-Bo4")]
    a.bar([0,1],vals,width=.52,color=[C["single"],C["dense"]])
    a.set(xticks=[0,1],xticklabels=["Original","Shared"],ylabel="Eager rollout mean (ms)",ylim=(0,305))
    x=next(row["median_ms"] for row in load("cuda-graph")["p5_precision"] if row["n"]==4)
    full=[x[path]["full_ms"] for path in ("eager","graph")]
    expert=[x[path]["expert_ms"] for path in ("eager","graph")]
    b.bar([0,1],expert,width=.59,color=C["dense"],label="Expert")
    b.bar([0,1],np.subtract(full,expert),bottom=expert,width=.59,color=C["light"],
          edgecolor=C["single"],linewidth=.35,label="Remainder")
    b.set(xticks=[0,1],xticklabels=["Eager","Graph"],ylabel="Fixed-query median (ms)",ylim=(0,280))
    b.legend(loc="upper right",ncol=1)
    prec=load("precision/summary")
    c.bar([0,1],[prec["original_bytes"]/1e6,prec["formats"]["int8"]["bytes"]/1e6],
          width=.58,color=[C["dense"],C["green"]])
    c.set(xticks=[0,1],xticklabels=["FP32","INT8"],ylabel="Stored critic (MB)",ylim=(0,195))
    for ax, l in zip(axes,"abc"):
        grid(ax)
        panel(ax,l)
    save(fig,"system-cost")


def q0_gate():
    data=rows("q0-by-query")
    fig,(ax,b)=plt.subplots(1,2,figsize=(WIDTH,1.7),layout="constrained")
    x=[int(r["query"]) for r in data]
    values=lambda key: [float(r[key]) for r in data]
    ax.plot(x,values("median"),color=C["dense"])
    ax.fill_between(x,values("q25"),values("q75"),color=C["pale_blue"])
    ax.set(ylabel="Candidate-zero score",xlabel="Query",xticks=[1,5,10,14])
    for name, color in (("gate25","dense"),("gate50","front"),("gate75","late")):
        b.plot(x,values(name),label=name[4:]+"th pct.",color=C[color])
    b.set(ylabel="Gate activation fraction",xlabel="Query",xticks=[1,5,10,14],ylim=(0,1.08))
    b.legend(loc="lower center",bbox_to_anchor=(.5,1.02),ncol=3,columnspacing=.6,
             handlelength=1.2,borderaxespad=.15)
    for q,l in ((ax,"a"),(b,"b")):
        panel(q,l)
        grid(q)
    save(fig,"q0-gate")


def graph_closed_loop():
    data=load("graph-closed-loop/summary")
    fig,(a,b)=plt.subplots(1,2,figsize=(WIDTH,1.78),layout="constrained")
    for label, tag, y in (("front5_minus_dense","front",1),("dense_minus_single","single",0)):
        v=data["paired"][label]
        sign=1 if tag=="front" else -1
        for ax,key,ci in ((a,"success_difference_pp","success_ci95_pp"),(b,"episode_ms_difference","episode_ms_ci95")):
            point=v[key]*sign
            limits=sorted(l*sign for l in v[ci])
            ax.errorbar(point,y,xerr=[[point-limits[0]],[limits[1]-point]],fmt=MARKERS[tag],
                        color=C[tag],ms=4,capsize=2.4)
    for ax,l in ((a,"a"),(b,"b")):
        ax.axvline(0,color=C["single"],lw=.65)
        ax.set(yticks=[0,1],yticklabels=["Single","EWS-5"],ylim=(-.58,1.58))
        panel(ax,l)
        grid(ax,"x")
    a.set(xlabel="Success difference vs. full (pp)")
    b.set(xlabel="Core time difference vs. full (ms)")
    save(fig,"graph-closed-loop")


def graph_query_cost():
    with (RESULTS/"graph-closed-loop"/"episodes.jsonl").open() as stream:
        ledger = [json.loads(line) for line in stream]
    fig, (a,b) = plt.subplots(1,2,figsize=(WIDTH,1.75),layout="constrained")
    for key, tag in (("single","single"),("dense","dense"),("front5","front")):
        episodes = [r["decision_ms"] for r in ledger if r["arm"] == key]
        x = np.arange(1,max(map(len,episodes))+1)
        reached = [[t[i-1] for t in episodes if len(t) >= i] for i in x]
        a.plot(x,[np.median(t) for t in reached],color=C[tag],marker=MARKERS[tag],
               ms=2.6,label=LABELS[tag],ls="--" if tag=="dense" else "-")
        b.plot(x,[len(t) for t in reached],color=C[tag],marker=MARKERS[tag],ms=2.6,
               ls="--" if tag=="dense" else "-")
    a.set(ylabel="Median core time (ms)")
    b.set(ylabel="Episodes reaching query")
    a.legend(loc="lower left",bbox_to_anchor=(0,1.02),ncol=3,columnspacing=.65,
             handlelength=1.4,borderaxespad=.15)
    for ax,l in ((a,"a"),(b,"b")):
        ax.set(xlabel="Reached query index",xticks=(1,5,10,14))
        grid(ax)
        panel(ax,l)
    save(fig,"graph-query-cost")


def lighting_diagnostic():
    screen=load("plus-light-decomposition/screen-summary")
    branch=load("plus-light-decomposition/branch-summary")
    fig,(a,b)=plt.subplots(1,2,figsize=(WIDTH,1.9),layout="constrained")
    for i,k in enumerate(("cc_vs_cl","lc_vs_ll","cc_vs_lc","cl_vs_ll")):
        v=screen["comparisons"][k]
        error(a,i,100*v["rate"],[100*z for z in v["paired_95ci"]],fmt="o",color=C["dense" if i<2 else "front"])
    a.set(ylabel="Winner change (%)",xticks=range(4),xticklabels=["cc→cl","lc→ll","cc→lc","cl→ll"],ylim=(0,49))
    for i,k in enumerate(("all","affected")):
        v=branch[k]
        error(b,i,100*v["light_minus_clean"],[100*z for z in v["paired_95ci"]],fmt="D",color=C["front"])
    b.axhline(0,color=C["single"],lw=.65)
    b.set(ylabel="Lighting − clean scoring (pp)",xticks=[0,1],xticklabels=["All matched","Winner changes"],ylim=(-37,6))
    for ax,l in ((a,"a"),(b,"b")):
        grid(ax)
        panel(ax,l)
    save(fig,"lighting-diagnostic")


def pp(value):
    return f"{value:+.2f}"


def markers(values):
    return "[" + ", ".join(f"{v:.2f}" for v in values) + "]"


def write_tables(data):
    lines = [
        r"% Generated by paper/make_submission.py",
        r"\begin{table}[t]\centering\small\setlength{\tabcolsep}{3.1pt}",
        r"\caption{Success rates (\%) by cohort. Bold marks the best observed rate within each row. The paired interval describes EWS minus full Bo4. Call savings use realized four-way batches.}",
        r"\label{tab:cohorts}",
        r"\begin{tabular}{@{}lrrrrrr@{}}\toprule",
        r"Cohort & Single & Full Bo4 & EWS & Late & EWS $-$ full (pp) & Calls saved \\",
        r"\midrule",
    ]
    readme=[]
    for name,key,selected,contrast,saved in data:
        sample=selected["single"][1]
        best=max(c/n for c,n,_ in selected.values())
        rates=[]
        for tag in ARMS:
            if tag not in selected:
                rates.append("--")
                continue
            c,n,_=selected[tag]
            v=f"{100*c/n:.2f}"
            rates.append(r"\textbf{"+v+"}" if c/n==best else v)
        effect=pp(contrast["pp"])+" "+markers(contrast["ci"])
        lines.append(" & ".join([f"{name} ({sample})",*rates,f"${effect}$",f"{saved:.1f}"+r"\%"])+r" \\")
        readme.append(f"| {name} | "+" | ".join(f"{selected[k][0]}/{selected[k][1]}" if k in selected else "—" for k in ARMS)+f" | {saved:.1f}% |")
    lines.extend([r"\bottomrule\end{tabular}",
                  r"\par\smallskip\footnotesize\textit{Note.} Late was unmeasured on Long Task2 and Plus lighting.",
                  r"\end{table}"])
    (PAPER/"tables/cohorts.tex").write_text("\n".join(lines)+"\n",encoding="ascii")
    path=PAPER.parent/"README.md"
    readme_text=path.read_text(encoding="utf-8")
    start,end="<!-- cohort-table:start -->","<!-- cohort-table:end -->"
    if start in readme_text:
        before,_,rest=readme_text.partition(start)
        _,_,after=rest.partition(end)
        table="| Cohort | Single | Full Bo4 | EWS | Late | Four-way calls saved |\n|---|---:|---:|---:|---:|---:|\n"
        path.write_text(before+start+"\n"+table+"\n".join(readme)+"\n"+end+after,encoding="utf-8")


def write_macros():
    archive=load("scheduled_selection/summary")
    graph=load("graph-closed-loop/summary")
    marg=load("scheduled_selection/marginal-value")["segments"][1]
    branch=load("scheduled_selection/critic-oracle")
    late=archive["object-combined"]["first_five_vs_late"]
    light=load("plus-light-decomposition/screen-summary")
    lp=load("plus-light-decomposition/branch-summary")["all"]
    pruning=load("structured-pruning/summary")
    arms=graph["arms"]
    gr=graph["paired"]["front5_minus_dense"]
    vals={
        "JLateGain": f'{late["pp"]:.2f}',
        "JLateCI": markers(late["ci"]),
        "JLateCalls": f'{archive["object-sweep"]["totals"]["late"]["calls"]+archive["object-fresh"]["totals"]["late"]["calls"]:,}',
        "JEarlyCalls": f'{archive["object-sweep"]["totals"]["front5"]["calls"]+archive["object-fresh"]["totals"]["front"]["calls"]:,}',
        "JCapture": f'{marg["capture_pct"]:.0f}',
        "JCallShare": f'{marg["full_call_share_pct"]:.1f}',
        "JGraphSaving": f'{100*(1-arms["front5"]["mean_episode_ms"]/arms["dense"]["mean_episode_ms"]):.1f}',
        "JGraphWins": f'{arms["front5"]["successes"]}/{arms["front5"]["episodes"]}',
        "JGraphCI": markers(gr["success_ci95_pp"]),
        "JGraphSingleCost": f'{graph["paired"]["front5_minus_single"]["episode_ms_difference"]:.2f}',
        "JGraphSingleCI": markers(graph["paired"]["front5_minus_single"]["episode_ms_ci95"]),
        "JGraphSavedMs": f'{abs(gr["episode_ms_difference"]):.2f}',
        "JGraphSavedMsCI": markers(sorted(abs(z) for z in gr["episode_ms_ci95"])),
        "JGraphFullCalls": f'{arms["dense"]["four_way_calls"]:,}',
        "JGraphEarlyCalls": f'{arms["front5"]["four_way_calls"]:,}',
        "JRankAUC": f'{branch["mixed_auc"]:.2f}',
        "JOracleGap": f'{branch["contrasts"]["oracle_minus_critic"]["difference_pp"]:.2f}',
        "JLightFlip": f'{light["comparisons"]["lc_vs_ll"]["flips"]}/{light["anchors"]}',
        "JLightDiff": f'{100*lp["light_minus_clean"]:.2f}',
        "JLightCI": markers([100*z for z in lp["paired_95ci"]]),
        "JLightBranchN": str(lp["anchors"]),
        "JLightBranchRuns": str(4*lp["anchors"]),
        "JLightCleanWins": str(int(lp["clean"])),
        "JLightWins": str(int(lp["light"])),
        "JLightRescues": str(lp["rescues"]),
        "JLightHarms": str(lp["harms"]),
        "JPruneOnly": "--".join(f'{r["arms"]["prune"]["p50_speedup_pct"]:.2f}' for r in reversed(pruning)),
        "JPruneCache": "--".join(f'{r["arms"]["prune_cache"]["p50_speedup_pct"]:.2f}' for r in reversed(pruning)),
    }
    out="% Generated by paper/make_submission.py\n"
    out+="".join(SLASH+"newcommand{"+SLASH+k+"}{"+v+"}\n" for k,v in vals.items())
    (PAPER/"submission-metrics.tex").write_text(out,encoding="ascii")


def main():
    data=list(cohort_data())
    write_tables(data)
    write_macros()
    teaser()
    method()
    marginal()
    cohorts(data)
    width_time()
    mechanism()
    system_fig()
    q0_gate()
    graph_closed_loop()
    graph_query_cost()
    lighting_diagnostic()


if __name__ == "__main__":
    main()
