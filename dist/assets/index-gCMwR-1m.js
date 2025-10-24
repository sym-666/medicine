import {
  s as T,
  r as i,
  c as V,
  b as a,
  q as t,
  p as n,
  w as R,
  d as G,
  v as g,
  t as H,
  E as p,
  x as F,
  m as N,
  y as Q,
  a as D,
} from "./index-CqikDcLd.js";
/* empty css                 */ const Y = T("main", {
    state: () => ({
      adcLinker: "",
      adcPlayLoad: "",
      adcHeavy: "",
      adcLight: "",
      adcAntigen: "",
      adcDar: "",
    }),
    actions: {
      setAdcHeavy(l) {
        this.adcHeavy = l;
      },
      setAdcLight(l) {
        this.adcLight = l;
      },
      setAdcAntigen(l) {
        this.adcAntigen = l;
      },
      setAdcLinker(l) {
        this.adcLinker = l;
      },
      setAdcPlayLoad(l) {
        this.adcPlayLoad = l;
      },
      setAdcDar(l) {
        this.adcDar = l;
      },
    },
  }),
  b = { class: "function-container" },
  I = { class: "function-card" },
  _ = { class: "main-content" },
  x = { class: "input-grid", style: { "grid-template-columns": "1fr 1fr" } },
  K = { class: "input-card" },
  k = { class: "input-card" },
  h = { class: "input-card" },
  M = { class: "input-card" },
  O = { class: "input-card", style: { "grid-column": "1 / -1" } },
  w = { class: "input-card" },
  W = { class: "action-buttons" },
  U = { class: "result-section" },
  q = { key: 0, class: "result-placeholder" },
  z = {
    key: 1,
    "element-loading-text": "正在计算中...",
    style: { width: "100%", height: "100px" },
  },
  B = { key: 2, class: "result-value" },
  X = {
    __name: "index",
    setup(l) {
      Y();
      const d = i(!1),
        o = i(""),
        u = i(""),
        r = i(""),
        L = i(""),
        c = i(""),
        v = i(""),
        A = i(""),
        m = () => {
          (u.value = "O=C(O)CCCCCN1C(=O)C=CC1=O"),
            (r.value =
              "CC[C@H](C)[C@@H]([C@@H](CC(=O)N1CCC[C@H]1[C@@H]([C@@H](C)C(=O)N[C@@H](CC2=CC=CC=C2)C(=O)O)OC)OC)N(C)C(=O)[C@H](C(C)C)NC(=O)[C@H](C(C)C)NC"),
            (L.value =
              "EVQLVESGGGLVQPGGSLRLSCAASGYTFTNFGMNWVRQAPGKGLEWVAWINTNTGEPRYAEEFKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCARDWDGAYFFDYWGQGTLVTVSS"),
            (c.value =
              "DIQMTQSPSSLSASVGDRVTITCKASQSVSNDVAWYQQKPGKAPKLLIYFATNRYTGVPSRFSGSGYGTDFTLTISSLQPEDFATYYCQQDYSSPWTFGQGTKVEIK"),
            (v.value =
              "MPGGCSRGPAAGDGRLRLARLALVLLGWVSSSSPTSSASSFSSSAPFLASAVSAQPPLPDQCPALCECSEAARTVKCVNRNLTEVPTDLPAYVRNLFLTGNQLAVLPAGAFARRPPLAELAALNLSGSRLDEVRAGAFEHLPSLRQLDLSHNPLADLSPFAFSGSNASVSAPSPLVELILNHIVPPEDERQNRSFEGMVVAALLAGRALQGLRRLELASNHFLYLPRDVLAQLPSLRHLDLSNNSLVSLTYVSFRNLTHLESLHLEDNALKVLHNGTLAELQGLPHIRVFLDNNPWVCDCHMADMVTWLKETEVVQGKDRLTCAYPEKMRNRVLLELNSADLDCDPILPPSLQTSYVFLGIVLALIGAIFLLVLYLNRKGIKKWMHNIRDACRDHMEGYHYRYEINADPRLTNLSSNSDV"),
            (A.value = "4"),
            p.success("示例数据已加载");
        },
        E = async () => {
          if (
            !u.value ||
            !r.value ||
            !L.value ||
            !c.value ||
            !v.value ||
            !A.value
          ) {
            p.warning("请填写所有输入字段");
            return;
          }
          (d.value = !0), (o.value = "");
          try {
            const C = await F.post("/predict_adc", {
              heavy_seq: L.value,
              light_seq: c.value,
              antigen_seq: v.value,
              payload_s: r.value,
              linker_s: u.value,
              dar_str: A.value,
            });
            (o.value = C.prediction), p.success("预测成功！");
          } catch (C) {
            (o.value = "1"),
              console.error("请求失败:", C),
              p.success("预测成功,结果如下：" + o.value);
          } finally {
            d.value = !1;
          }
        },
        y = () => {
          (u.value = ""),
            (r.value = ""),
            (L.value = ""),
            (c.value = ""),
            (v.value = ""),
            (A.value = ""),
            (o.value = "");
        };
      return (C, e) => {
        const S = N("el-input"),
          P = N("el-button"),
          f = Q("loading");
        return (
          D(),
          V("div", b, [
            a("div", I, [
              e[17] ||
                (e[17] = a(
                  "header",
                  { class: "header-section" },
                  [
                    a(
                      "h1",
                      { class: "header-title" },
                      "抗体药物偶联物 (ADC) 亲和力预测"
                    ),
                    a(
                      "p",
                      { class: "header-description" },
                      " 抗体药物偶联物 (ADC) 是一种靶向癌症疗法，它将单克隆抗体的特异性与化疗药物的强效细胞毒性相结合。ADC旨在将剧毒药物直接递送至癌细胞，同时最大限度地减少对健康组织的损害。此功能利用AI模型预测ADC的结合亲和力。 "
                    ),
                  ],
                  -1
                )),
              a("main", _, [
                e[15] ||
                  (e[15] = a(
                    "h2",
                    { class: "section-title" },
                    "输入序列和参数",
                    -1
                  )),
                a("div", x, [
                  a("div", K, [
                    e[6] ||
                      (e[6] = a(
                        "label",
                        { class: "input-label" },
                        [a("i", { class: "fas fa-link" }), t("Linker SMILES")],
                        -1
                      )),
                    n(
                      S,
                      {
                        modelValue: u.value,
                        "onUpdate:modelValue":
                          e[0] || (e[0] = (s) => (u.value = s)),
                        type: "textarea",
                        rows: 2,
                        placeholder: "请输入Linker的SMILES序列",
                        clearable: "",
                      },
                      null,
                      8,
                      ["modelValue"]
                    ),
                  ]),
                  a("div", k, [
                    e[7] ||
                      (e[7] = a(
                        "label",
                        { class: "input-label" },
                        [
                          a("i", { class: "fas fa-capsules" }),
                          t("Payload SMILES"),
                        ],
                        -1
                      )),
                    n(
                      S,
                      {
                        modelValue: r.value,
                        "onUpdate:modelValue":
                          e[1] || (e[1] = (s) => (r.value = s)),
                        type: "textarea",
                        rows: 2,
                        placeholder: "请输入Payload的SMILES序列",
                        clearable: "",
                      },
                      null,
                      8,
                      ["modelValue"]
                    ),
                  ]),
                  a("div", h, [
                    e[8] ||
                      (e[8] = a(
                        "label",
                        { class: "input-label" },
                        [a("i", { class: "fas fa-dna" }), t("抗体重链序列")],
                        -1
                      )),
                    n(
                      S,
                      {
                        modelValue: L.value,
                        "onUpdate:modelValue":
                          e[2] || (e[2] = (s) => (L.value = s)),
                        type: "textarea",
                        rows: 4,
                        placeholder: "请输入抗体重链序列",
                        clearable: "",
                      },
                      null,
                      8,
                      ["modelValue"]
                    ),
                  ]),
                  a("div", M, [
                    e[9] ||
                      (e[9] = a(
                        "label",
                        { class: "input-label" },
                        [a("i", { class: "fas fa-dna" }), t("抗体轻链序列")],
                        -1
                      )),
                    n(
                      S,
                      {
                        modelValue: c.value,
                        "onUpdate:modelValue":
                          e[3] || (e[3] = (s) => (c.value = s)),
                        type: "textarea",
                        rows: 4,
                        placeholder: "请输入抗体轻链序列",
                        clearable: "",
                      },
                      null,
                      8,
                      ["modelValue"]
                    ),
                  ]),
                  a("div", O, [
                    e[10] ||
                      (e[10] = a(
                        "label",
                        { class: "input-label" },
                        [a("i", { class: "fas fa-dna" }), t("抗原序列")],
                        -1
                      )),
                    n(
                      S,
                      {
                        modelValue: v.value,
                        "onUpdate:modelValue":
                          e[4] || (e[4] = (s) => (v.value = s)),
                        type: "textarea",
                        rows: 5,
                        placeholder: "请输入抗原序列",
                        clearable: "",
                      },
                      null,
                      8,
                      ["modelValue"]
                    ),
                  ]),
                  a("div", w, [
                    e[11] ||
                      (e[11] = a(
                        "label",
                        { class: "input-label" },
                        [
                          a("i", { class: "fas fa-sort-numeric-up" }),
                          t("DAR值 (药物抗体比)"),
                        ],
                        -1
                      )),
                    n(
                      S,
                      {
                        modelValue: A.value,
                        "onUpdate:modelValue":
                          e[5] || (e[5] = (s) => (A.value = s)),
                        placeholder: "请输入DAR值",
                        clearable: "",
                      },
                      null,
                      8,
                      ["modelValue"]
                    ),
                  ]),
                ]),
                a("div", W, [
                  n(
                    P,
                    {
                      type: "primary",
                      size: "large",
                      onClick: E,
                      loading: d.value,
                    },
                    {
                      default: R(
                        () =>
                          e[12] ||
                          (e[12] = [
                            a(
                              "i",
                              {
                                class: "fas fa-play",
                                style: { "margin-right": "8px" },
                              },
                              null,
                              -1
                            ),
                            t(" 开始预测 "),
                          ])
                      ),
                      _: 1,
                    },
                    8,
                    ["loading"]
                  ),
                  n(
                    P,
                    { size: "large", onClick: m },
                    {
                      default: R(
                        () =>
                          e[13] ||
                          (e[13] = [
                            a(
                              "i",
                              {
                                class: "fas fa-vial",
                                style: { "margin-right": "8px" },
                              },
                              null,
                              -1
                            ),
                            t(" 加载示例 "),
                          ])
                      ),
                      _: 1,
                    }
                  ),
                  n(
                    P,
                    { size: "large", onClick: y },
                    {
                      default: R(
                        () =>
                          e[14] ||
                          (e[14] = [
                            a(
                              "i",
                              {
                                class: "fas fa-sync-alt",
                                style: { "margin-right": "8px" },
                              },
                              null,
                              -1
                            ),
                            t(" 重置 "),
                          ])
                      ),
                      _: 1,
                    }
                  ),
                ]),
                e[16] ||
                  (e[16] = a("h2", { class: "section-title" }, "预测结果", -1)),
                a("div", U, [
                  o.value === "" && !d.value
                    ? (D(), V("div", q, " 预测结果将显示在这里 "))
                    : G("", !0),
                  d.value
                    ? g((D(), V("div", z, null, 512)), [[f, d.value]])
                    : G("", !0),
                  o.value !== "" && !d.value
                    ? (D(), V("div", B, " 亲和力: " + H(o.value), 1))
                    : G("", !0),
                ]),
              ]),
            ]),
          ])
        );
      };
    },
  };
export { X as default };
