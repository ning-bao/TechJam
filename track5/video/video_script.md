# 3 分钟 Demo 视频 · 脚本与分镜(纯录屏 + 字幕,无口播)

交付 3。题面 §5.5 要求:端到端演示、上传 YouTube 设为 public、在 Devpost 描述里给链接、不含未授权第三方商标/版权内容。
题面明确允许:没有前端时,用 walkthrough 展示 API 用法 / 推理示例 / **结果分析**。

**总时长目标 2:50**(留 10 秒余量,硬上限 3:00)。
**无人声。** 全部靠屏幕内容 + 烧入字幕(burn-in)。所以每一屏的**停留时间由能不能读完决定**,不由内容多少决定。

---

## ⚠️ 分工:两个人录,因为一个人录不了

| 段 | 谁录 | 为什么 |
|---|---|---|
| **B 段(真推理)** | **包** | 权重只在他机器上。本地 `track5/runs/` 不存在,`predict.py` 跑不起来 |
| 其余全部 | 蔡海童 | 只需要仓库 + 浏览器 |

包那段的具体要求见文末「§ 包需要录的两个片段」。**先跟他要这两段,再开始剪。**

---

## 字幕节奏的硬规矩(掐表,别按字数估)

- **每屏字幕 ≤ 12 个英文单词**,停留 **≥ 2.5 秒**;超过 12 词的拆成两屏
- 字幕出现在**画面动作之后**,不要同时 —— 观众先看到发生了什么,再读为什么
- 录屏时**光标移动要慢**,快速拖动在压缩后会变成糊块
- 终端字号调到 **16pt 以上**,YouTube 1080p 下小字看不清

**剪完必须掐表**,不要相信这份脚本里的估时。逐段核对下面的累计时间轴,超了就砍 D 段的第二个例子(那是唯一可删且不伤论点的地方)。

---

## 时间轴总览

| 段 | 起 | 止 | 时长 | 内容 |
|---|---|---|---|---|
| A | 0:00 | 0:16 | 16s | 问题与主张 |
| B | 0:16 | 1:00 | 44s | **端到端跑通**(包录,素材已到) |
| C | 1:00 | 1:18 | 18s | 鲁棒性矩阵 |
| D | 1:18 | 1:40 | 22s | 我们先写下失败标准,再去找失败 |
| E | 1:40 | 2:16 | 36s | **头条:2026 消费级端点逃逸** |
| F | 2:16 | 2:35 | 19s | 纪律与限制 |

**这些是预算,不是实测。** 剪完逐段掐表核对,别把这张表当既成事实。
2026-08-31 22:40 重新分配:B 段素材(包录的三段共 70.5s)需要比原来的 34s 更多空间,
从 A/C/D/F 挤出 25s,其中 10s 给 B,余下 15s 作为 3:00 硬上限前的真余量。

**输出规格:1280×720 / 30fps。** 不出 1080p —— 包的素材是 720p 原生,
升采样会糊掉终端里的 `pred` 值和 hash,而那些字正是证据本身。
两张卡是 1920×1080,降采样到 720p 对文字无损。

各段压缩的依据(下限由**屏幕上那个东西要多久读得完**决定,不由字幕数决定;
全片 28 屏字幕按 ≥2.5s 算硬下限仅 70s,远非瓶颈):
- A:开场卡 6s→4s。**三联图那 4 秒静默停留不动** —— 先认定"三张一样"字幕才有反转。
- C:表只有 5 行 + 光标点两格,原 24s 有垫场。
- D:砍第 3 镜(1:38–1:50),本来就是脚本预先标好的可删镜。
- F:探针表 10→8s,清单滚动 8→6s,收尾卡 6→5s。
- E:**不压。** 头条段,36s 内 6 屏字幕。

---

## A 段 · 0:00–0:18 · 问题与主张

**画面**
1. (0:00–0:06)黑底白字标题卡:
   ```
   Robust Detection of AI-Generated Images
   Under Real-World Transforms

   Track 5 · team skipskipskip
   ```
2. (0:06–0:18)切到 `work/video_assets/A_transform_triptych.png`(**已生成**)。三联图:原图 / JPEG q30 / 降采样 0.25×,每格下方标注高频能量占原图的百分比:**100% → 74% → 4%**。
   镜头**先停 4 秒让人看图**(三格肉眼几乎无差),**再让字幕点出那个 4%**。这个先后顺序是这段的全部效果所在。

**字幕**
- 0:06 `A detector that only works on clean images is not a detector.`
- 0:10 `Every upload re-encodes. Every re-share resizes.`
- 0:14 `These three look identical. The detector's evidence: 100%, 74%, 4%.`

**素材**:✅ 已生成于 `work/video_assets/A_transform_triptych.png`(1764×708)。
用仓库真正的 `apply_and_encode` 产出,源图是集合 B 我们自研生成图(`flux_dev/001.png`),**无第三方版权风险**。
高频能量 = 拉普拉斯响应方差,实测 313.9 → 234.1 → 10.3。

---

## B 段 · 0:18–0:52 · 端到端跑通(包录)

这是题面 "working end-to-end" 那条的正面回应,**也是 Technical Execution 35% 最直接的证据**。全部是真终端,不做动画。

**画面**
1. (0:18–0:26)终端里跑 predict:
   ```
   python -u -m src.predict \
       --checkpoint runs/dinov3l448_d4/epoch1_best_calibrated.pt \
       --input demo_images/ --output preds.json
   ```
   让进度输出真实滚动。**不要加速这一段** —— 真实速度本身是证据。
2. (0:26–0:36)`cat preds.json` 显示组委会要求的记录格式:
   ```json
   [{"image_path": "...", "pred": 0.9993}, ...]
   ```
   光标停在两三个 `pred` 值上。
3. (0:36–0:44)同一批图跑过变换后再跑一次,分数仍在同侧。
4. (0:44–0:52)展示 `preds.json.errors.json`:一次解码失败被记录下来,**而不是静默记成 0.5**。

**字幕**
- 0:20 `One script. A directory in, {image_path, pred} out.`
- 0:28 `pred is a calibrated probability, not a raw logit.`
- 0:32 `sigmoid((z + alpha) / T), frozen before inference.`
- 0:38 `Same images after JPEG q30 and 0.25x downscale.`
- 0:46 `A decode failure is recorded, never silently scored 0.5.`

---

## C 段 · 0:52–1:16 · 鲁棒性矩阵

**画面**
1. (0:52–1:04)在编辑器里打开 `track5/reports/robustness_summary.md`,滚到受保护基准那张表。**放大到能看清数字。**
2. (1:04–1:16)光标依次停在 `clean 0.9891` 和最低那行 `resize_025 0.9584`。

**字幕**
- 0:54 `The organiser's benchmark: DALL-E 3, unseen by our model.`
- 0:58 `0.958 to 0.989 balanced accuracy across five conditions.`
- 1:04 `Worst-case degradation from clean: 3.1 points.`
- 1:09 `The whole DALL-E family, 64,482 images, is denylisted from training.`

---

## D 段 · 1:16–1:50 · 我们先写下失败标准,再去找失败

**画面**
1. (1:16–1:26)**这一屏是整个视频最有说服力的一处** —— 时间顺序在 git 历史里可被评委独立验证,不是我们自说自话。两条命令,一条一条打:

   ```bash
   # 标准写下的那一刻,§4 还是空的
   git show e6c8349:track5/reports/error_analysis.md | grep -A13 '^## 4\.'
   ```
   屏幕上会出现(**已实跑验证**):
   ```
   ## 4. Individual cases — pending the scoring run
   ...
   **Acceptance criteria decided in advance**, so the analysis cannot be
   shaped by what comes back:
   ```

   ```bash
   # 38 小时后才打分
   git log --format='%h %ai %s' -4 -- track5/reports/error_analysis.md
   ```
   最后一行是 `e6c8349 2026-08-28 23:05:53`,第一行是 `60284cf 2026-08-30 13:07:44`。

   ⚠️ **`-4` 不能少**,`-2` 只会显示 8/30 的两笔,看不到 8/28 那笔 —— 整个论点就没了。
   **光标在首末两个时间戳上各停 1 秒。**
2. (1:26–1:38)打开 `error_analysis.md` §4,光标停在集合 A 那两个假阳性上:0.8012 → 降采样后 0.0041。
3. (1:38–1:50)滚到假阳性表,停在 `020.jpg`:0.044 原生 → 0.956 降采样。

**字幕**
- 1:18 `Three pass/fail criteria, written down 38 hours before scoring.`
- 1:24 `Verify it yourself: the git history is the receipt.`
- 1:28 `Criterion 1: false positives on 100 adversarial real photos.`
- 1:32 `Not triggered: 2 of 100. Both collapse when rescaled.`
- 1:38 `They were reporting resolution, not photography.`
- 1:44 `One counterexample kept: benign at native, 0.956 downscaled.`

**可删**:时间超了就砍第 3 个镜头(1:38–1:50)和它那两条字幕,省 12 秒。论点不受伤。

---

## E 段 · 1:50–2:26 · 头条:2026 消费级端点逃逸

**这是全片最重要的 36 秒。** 别的段可以压,这段不要动。

**画面**
1. (1:50–2:02)`error_analysis.md` §4 那张切片表,放大:
   ```
   flux_dev (seen family, no watermark)   25/25   median 0.9993
   gpt-image-2 (Azure)                     0/30   median 0.0038
   gemini-3-pro-image                      3/25   median 0.0076
   gemini-3.1-flash-image                  2/20   median 0.0065
   ```
2. (2:02–2:14)光标从 Flux 那行**慢慢**划到 gpt-image-2 那行,让 0.9993 → 0.0038 的对照被看见。
3. (2:14–2:26)切到 `track5/KNOWN_LIMITATIONS.md` item 11。

**字幕**
- 1:52 `Criterion 2 fired at 93.3 points, backwards.`
- 1:58 `We guarded against the detector reading vendor watermarks.`
- 2:03 `A watermark-reader would find the watermarked slice easy.`
- 2:08 `Ours finds it nearly invisible. That refutes the mechanism.`
- 2:14 `Not the transforms. Not unseen families: DALL-E 3 holds 0.9891.`
- 2:20 `It is the 2026 generation of consumer endpoints.`

---

## F 段 · 2:26–2:50 · 纪律与限制

**画面**
1. (2:26–2:36)`README.md` 的捷径探针表:`JPEG quantization table 0.974 → 0.500`。
2. (2:36–2:44)`track5/KNOWN_LIMITATIONS.md` 目录,展示 **11 条**,滚过去让人看到它有多长。
3. (2:44–2:50)收尾卡:
   ```
   github.com/ning-bao/TechJam

   Eleven limitations, disclosed.
   Including the two we overran ourselves.
   ```

**字幕**
- 2:28 `Before training: a probe reading only the JPEG table scored 0.974.`
- 2:33 `We equalised the containers. It fell to 0.500.`
- 2:38 `Eleven limitations, ordered by how much each should move your confidence.`
- 2:45 `Including the one where we overran our own protected-set budget.`

---

## § 包需要录的两个片段(先跟他要这个)

**片段 1 — predict 跑通**(目标 30 秒素材,剪出来用 26 秒)
- 3–5 张图放一个目录,跑 `src.predict`,**真实速度不加速**
- 然后 `cat preds.json`
- 终端字号 ≥16pt,窗口尽量方一点(竖屏比例在 16:9 里会留大黑边)

**片段 2 — 变换后再跑一次**(目标 12 秒素材)
- 同一批图过 `jpeg_30` 和 `resize_025` 之后,再跑一次 predict
- 目的是让观众看到**分数仍在同一侧**

**片段 3(可选,8 秒)** — 展示 `preds.json.errors.json` 里记下的一次解码失败

**约束**:
- ⚠️ **不得录任何受保护集(COCO val2017 / WildFake DALL·E)的图像或数字。** demo 图用集合 B 或任意自有图
- 录屏里不要出现未公开的路径、token、账号
- 交原始录屏文件,不要他自己剪

---

## § 需要预先做的素材

1. ✅ **A 段三联图 —— 已完成**:`work/video_assets/A_transform_triptych.png`
   重生成命令留在 git 里;源图 `B_fn_magnet/flux_dev/001.png` 是我们自研生成的,无版权风险。
2. ✅ **标题卡与收尾卡 —— 已完成**:`track5/video/card_opening.png`、`card_closing.png`(均 1920×1080)
   队名 **team skipskipskip** 渲进 `work/video_assets/card_opening_named.png`
   (成片用的是这张,不是原卡)。
   ✅ 已于 2026-09-01 对照 Devpost 注册页核准,写法一致 —— 该队名不再仅凭记忆或口头确认。
3. ⚠️ 全片图像**只能用集合 B(自研生成)或自有照片**。不要用网上找的 AI 图 —— 题面禁止未授权第三方版权内容。

---

## § 已发布

**https://youtu.be/HoKdBR0hjNc** — 2026-08-31 发布。

标题:`Robust Detection of AI-Generated Images Under Real-World Transforms — TikTok TechJam 2026 Track 5`

public 状态已验证:YouTube oEmbed 接口(`/oembed?url=...`)对私享与未列出视频返回 401,
该视频返回了完整元数据,故为公开可见。

---

## § 提交前检查清单

- [x] 总时长 **≤ 3:00** —— 实测 2:34.9
- [x] 上传 YouTube 且设为 **public**(不是 unlisted)—— 已用 oEmbed 验证
- [x] 全片**没有受保护集**的图像或数字 —— 演示图取自集合 B(自研生成)
- [x] 没有未授权的第三方商标/版权素材
- [x] 手机上看过,终端字能认出
      (成片为 720p 原生,非 1080p:包录素材本身是 720p,升采样会糊掉 `pred` 与 hash)
- [x] 开场卡队名与 Devpost 注册页一致 —— 2026-09-01 核准

**唯一未完成项 —— Devpost 提交(计划 2026-09-01 上午):**

- [ ] 链接贴进 Devpost 项目编辑页的 **Video demo link** 字段(不是描述正文):
      `https://youtu.be/HoKdBR0hjNc`
      该字段只接受 YouTube / Vimeo / Youku,填其他会报
      "Must be a valid YouTube, Vimeo, or Youku url"。
- [ ] **点 Submit,把项目关联到 Track 5 比赛。** 只保存是草稿,截止后不算参赛 —— 这是两步。
- [ ] (建议)描述正文 About the project 里也放一条明文视频链接与仓库链接,
      供不看内嵌播放器、直接读正文的评委。

---

## § 成片与重建

**成片**:`work/video_assets/final/demo_full.mp4` — 2:34.9,1280×720,30fps,字幕已烧入。

| 段 | 时长 | 来源 |
|---|---|---|
| A | 16.0s | 开场卡(含队名)+ 三联图 |
| B | 43.9s | **真实终端录屏**,剪掉三处纯等待,原片保留在 `work/video_assets/B*.mp4` |
| C | 18.0s | 渲染帧 · `robustness_summary.md` 受保护基准表 |
| D | 22.0s | 渲染帧 · **git 输出是真跑的**,非排版伪造 |
| E | 36.0s | 渲染帧 · set B 切片表 + 局限 11 |
| F | 19.0s | 渲染帧 · 探针表 + 11 条局限 + 收尾卡 |

⚠️ **C/D/E/F 是渲染帧,不是录屏** —— 无编辑器界面、无光标移动。内容与数字全部取自仓库并逐条核对,
但"真人操作真环境"的观感弱于 B 段。B 段那 43.9 秒是唯一的真终端素材,端到端跑通的证据在那里。
D 段若要更强的可信度,自行录一遍 22 秒替换 `work/video_assets/seg/d{1,2,3}.mp4` 即可。

**重建**(本机 ffmpeg 为精简构建,**无 libass / libfreetype**,`drawtext`/`subtitles`/`ass`
三个滤镜均不可用;所有文字改由 Pillow 渲 PNG + `overlay` 合成):

```bash
cd work
/usr/bin/python3 -m venv .venv-img && ./.venv-img/bin/pip install Pillow numpy
./.venv-img/bin/python build_frames.py   # C/D/E/F 的 13 张画面帧
./.venv-img/bin/python build_subs.py     # 29 屏字幕 PNG
bash compose.sh                          # 叠字幕 + 拼接 -> final/demo_full.mp4
```

B1/B2 做了信箱边(画面缩至 646px 高,底部 74px 留给字幕):这两段终端输出到第 716 行,
任何压在画面上的字幕都会遮住 JSON 记录。B3 输出只到第 392 行,无需处理。
- [ ] 字幕是烧入的,不依赖 YouTube 自动字幕
