# Izvoria hardware: specs, renting, buying, and client setup

Written 2026-10-14. The requirements come from this repository's configuration and the 2026-10-10 RTX 3090 measurement run (`docs/web_deploy.md`).
Prices are what could be found online in October 2026. GPU and RAM prices have risen sharply in 2026, so check them again before quoting anyone.
**est.** marks a rough estimate rather than a looked-up price. Every speed except the 3090's is an estimate scaled from the 3090 measurement.

**The rule for every tier: chat and ingestion run at the same time.** That needs **24 GB of VRAM on one card**.
- Measured on the 3090: the live side (answer model for 6 users, search models) took 12.8 GB, and with a video being ingested beside it the peak was **20.2 GB**.
- A 12–16 GB card cannot do this: chat would have to stop for every ingestion.
- Two 12 GB cards don't help yet either. The code can't place models on a second GPU, so it would sit idle.

---

## 1. What separates minimum from optimal: VRAM speed

Every tier has 24 GB or more; what differs is how fast that memory is.
- **Answer speed** (tokens per second while the model writes) scales almost directly with **memory bandwidth**.
- **Prompt reading** (long questions, reports over 100K tokens of evidence) and **ingestion** (Whisper, the video model) scale with the card's **compute**. Newer generations gain more there than on bandwidth.

| Card | VRAM | Bandwidth | Power | Answer speed, 1 user (est., 3090 = 97 tok/s measured) | Tier |
|---|---|---|---|---|---|
| RTX PRO 4000 Blackwell | 24 GB ECC | 672 GB/s | 140 W | ~70 tok/s | Minimum |
| RTX A5000 | 24 GB ECC | 768 GB/s | 230 W | ~80 tok/s | Minimum |
| **RTX 3090** | 24 GB | 936 GB/s | 350 W | **97 tok/s (measured)** | **Recommended (the measured baseline)** |
| RTX 3090 Ti | 24 GB | 1,008 GB/s | 450 W | ~105 tok/s | Recommended |
| RTX 4090 | 24 GB | 1,008 GB/s | 450 W | ~105 tok/s; prompt reading and ingestion ~1.5–2× faster | Recommended+ |
| RTX PRO 5000 Blackwell | 48 GB ECC | 1,344 GB/s | 300 W | ~140 tok/s | Optimal |
| RTX 5090 | 32 GB | 1,792 GB/s | 575 W | ~185 tok/s | Optimal |
| RTX PRO 6000 Blackwell | 96 GB ECC | 1,792 GB/s | 600 W | ~185 tok/s | Optimal (headroom for many firms) |

Avoid these 24 GB cards:
- **L4** (300 GB/s) and **RTX 4500 Ada** (432 GB/s): chat feels slow.
- **A10** (600 GB/s): a passively cooled server card that needs server airflow.
- **AMD / Intel cards:** the stack is CUDA-only.

**The cheapest card that qualifies is a recommended-tier one.** A used RTX 3090 (~€1,200) costs less than either minimum-tier card, and is faster.
- The minimum-tier cards earn their place differently: low power and noise, ECC memory, a warranty, a single slot.
- That makes them the right choice for a client's office appliance. For your own hosted server, the 3090 is the better choice.

---

## 2. Full machine per tier

**Two machine roles:**
- **Hosted GPU box** (yours): it keeps no client data at rest; the data stays on the VPS. It needs disk only for the models and scratch.
- **On-premise appliance** (a client's): it holds all their data, so it needs bigger, mirrored storage and a UPS.

| | **Minimum** | **Recommended** | **Optimal** |
|---|---|---|---|
| **GPU** | RTX PRO 4000 Blackwell 24 GB or RTX A5000 24 GB | RTX 3090 / 3090 Ti / 4090 24 GB | RTX 5090 32 GB, RTX PRO 5000 48 GB or RTX PRO 6000 96 GB |
| **CPU** | 8 cores / 16 threads (Ryzen 7 5700X class) | 12 cores (Ryzen 9 5900X / 7900 class) | 16 cores+ (Ryzen 9 9950X, or Threadripper for ECC and a 2nd GPU later) |
| **Motherboard** | B550: one PCIe 4.0 x16 slot, 2× M.2 | B550 (AM4) or B650 (AM5): PCIe 4.0/5.0 x16 with room for a 3-slot card, 2–3× M.2 | X670E / TRX50: PCIe 5.0, 3–4× M.2, room for two GPUs |
| **RAM** | 64 GB DDR4-3200 | 64 GB (DDR4 on AM4 is far cheaper than DDR5 right now) | 128 GB, ECC on Threadripper |
| **Storage, hosted box** | 1 TB NVMe | 1–2 TB NVMe | 2 TB NVMe |
| **Storage, on-premise** | 2 TB NVMe | 2× 2 TB NVMe mirrored | 2× 4 TB NVMe mirrored + backup disk |
| **PSU** | 650–750 W Gold | 850 W Gold (3090), 1000 W ATX 3.1 (4090) | 1200–1600 W ATX 3.1 |
| **UPS (on-premise)** | 1000 VA | 1500 VA | 2000 VA+ |

**Why these numbers:**
- **CPU:** video decoding, OCR on scanned pages (CPU only), sentence splitting and Postgres all run on the CPU, while the GPU serves chat.
- **RAM:**
  - Three processes are resident at once: the API with the search models, `llama-server` with the 5.7 GB answer model, and Postgres.
  - On top of that, the ingestion process loads each model through RAM and holds audio and video frames.
  - The 3090 box ran in 26 GB, but tightly. 64 GB is the comfortable minimum.
- **Disk:** the models take ~51 GB (~56 GB with Serbian). On-premise, every client recording, document and its index adds to that.
- **Network:** the hosted box only dials out, so it needs no open ports and no static IP. An office line works.

---

## 3. Costs to buy (October 2026)

Prices are in euros, from the sources below. Parts other than the GPU are rough estimates.

| | **Minimum** | **Recommended** | **Optimal** |
|---|---|---|---|
| GPU | RTX PRO 4000 Blackwell new: ~€1,700–3,300 (US $3,300; one German eBay listing €1,694) | **Used RTX 3090: ~€1,100–1,300** (eBay sold median €1,223); used RTX 4090: ~€2,100–2,400; new RTX 4090: ~€2,300–3,300 | RTX 5090: €4,600–6,800 in the EU now (inflated); RTX PRO 6000: ~$13–16k |
| CPU | est. €150–200 | est. €250–400 | est. €500–1,500 |
| Motherboard | est. €120–180 | est. €150–300 | est. €350–900 |
| RAM | 64 GB DDR4, est. €200–350 | 64 GB DDR4 est. €200–350, or DDR5 €550–900 | 128 GB, est. €1,100–2,000 |
| NVMe | est. €100–300 | est. €200–500 | est. €600–1,000 |
| PSU, case, cooler | est. €250 | est. €350–450 | est. €600 |
| UPS (on-premise only) | est. €150 | est. €250 | est. €400 |
| **Total** | **~€2,700–4,700** | **~€2,400–3,300 with a used 3090** (~€3,700–5,500 with a 4090) | **~€8,000–10,000 with a 5090**, ~€17,000+ with an RTX PRO 6000 |

**For your own hosted server:** the recommended build with a used 3090 on AM4 (Ryzen 9 5900X class, 64 GB DDR4) costs about **€2,500**. It is exactly what was measured.
- **Running cost:** mostly electricity, est. 150–450 W depending on load.
- **Payback:** about 12 months against a €214/month rented server.
- **Downside:** you are the repair service.

**For clients:** sell new cards with a warranty: an RTX PRO 4000 Blackwell (minimum, quiet, 140 W) or a new 4090 (recommended).

---

## 4. Renting (hosted version)

All of these have 24 GB, so all of them run chat and ingestion side by side.

| Option | Card (tier) | Price | Catch |
|---|---|---|---|
| HOSTKEY vgpu.v2-a5000 (VPS) | RTX A5000 (minimum) | €179/mo | *Pre-Order*. HOSTKEY (Amsterdam) also has sites in Russia, so get the EU location in writing |
| **Hetzner GEX45** | RTX PRO 4000 Blackwell (minimum) | ~€214/mo + ~€209 setup (trade-press figure; confirm on Hetzner's order page) | Same company as the VPS, under a German contract with a data-processing agreement. A dedicated machine nobody else can take. Helsinki. About 70% of the 3090's answer speed |
| HOSTKEY 3090 / A5000 dedicated | 3090 (recommended) / A5000 | €279–330/mo | *Pre-Order* |
| HOSTKEY 4090 dedicated | RTX 4090 (recommended+) | €360+/mo | Available now |
| Vast.ai on-demand, kept running | RTX 3090 (recommended) | ~$0.35/hr, ~$255/mo if never stopped | Not your hardware; a 17-day to 2.5-month maximum per instance |
| Vast.ai stop/start | RTX 3090 | ~$0.35/hr used + ~$20/mo disk each | A stopped instance's GPU can be rented by someone else (it happened 2026-10-14). Several stopped instances make it likelier that one can start |

**Nothing qualifying costs €100–150 a month.** The cheapest 24 GB rentals are HOSTKEY's A5000 VPS at €179 (not in stock) and Hetzner's GEX45 at ~€214.

**Buying the recommended build (~€2,500) costs about the same as a year of renting**, and it is faster than the GEX45.

---

## 5. A client buying its own on-premise appliance: what to charge

There is no market data for this here; this is a suggested structure, built from the work it takes.

### What a setup involves

1. **Hardware:** sourcing, assembly, a 24–48 h burn-in test.
2. **Install:** the OS, NVIDIA drivers, Postgres, the app, ~51 GB of models, the desktop app.
3. **The firm's glossary:** the profile interview, the Claude prompt (`docs/glossary_authoring/`), validation and review. Accuracy depends on it more than on anything else.
4. **First ingestion of their archive:** machine time, so plan for it to run overnight or for days. The 3090 indexed a 16-minute lecture in ~4 minutes while 3 users chatted.
5. **Training:** staff learn Chat, Search, Report and how to verify citations.
6. **Acceptance:** you and they ask questions over their own material and check the answers together.

That's realistically **3–5 working days** of your time per client, plus the machine time.

### Suggested quote

| Item | Small firm (minimum tier) | Most firms (recommended) | Large firm / hospital (optimal) |
|---|---|---|---|
| Appliance hardware, new, sold at cost + 15–25% | ~€3,500–5,500 | ~€4,500–6,500 (new 4090) | ~€10,000–20,000 |
| **Setup fee** (steps 1–6) | **€2,000–3,500** | **€2,500–4,000** | **€4,000–7,000** |
| Annual support & updates | 15–20% of the total a year, or a monthly fee | same | same |
| **One-time total** | **~€5,500–9,000** | **~€7,000–10,500** | **~€14,000–27,000** |

Things to settle in the contract:
- **Who owns the hardware.** The client usually does; then a hardware fault is a paid callout unless support covers it.
- **Remote access.** The appliance has no internet by design, so updates happen on site or through a connection the client opens for the purpose. Price on-site visits separately.

---

## Sources

**GPU rental:**
- Hetzner: [GPU servers](https://www.hetzner.com/dedicated-rootserver/matrix-gpu/), [GEX45 press release](https://www.hetzner.com/pressroom/hetzner-expands-its-gpu-portfolio-with-the-gex45/), [GEX45 pricing (IT Brief)](https://itbrief.co.uk/story/hetzner-launches-gex45-gpu-server-for-ai-workloads)
- HOSTKEY: [GPU servers](https://hostkey.com/gpu-dedicated-servers/), [NVIDIA servers](https://hostkey.com/dedicated-servers/rent-nvidia-servers/)
- Vast.ai: [managing instances](https://docs.vast.ai/guides/instances/manage-instances), [instance types](https://docs.vast.ai/guides/instances/choosing/instance-types)

**GPU prices:**
- RTX 3090 used: [pcprice.watch](https://www.pcprice.watch/search/RTX3090/DE/GPU), [bestvaluegpu](https://bestvaluegpu.com/en-eu/history/new-and-used-rtx-3090-price-history-and-specs/)
- RTX 4090: [pcprice.watch](https://www.pcprice.watch/gpu-buying-guide/rtx-4090-price-used-and-specs), [bestvaluegpu](https://bestvaluegpu.com/en-eu/history/new-and-used-rtx-4090-price-history-and-specs/), [gpuprix](https://gpuprix.com/eu/gpus/geforce-rtx-4090)
- RTX A5000: [rigprice](https://rigprice.com/gpu/rtx-a5000/), [gpudojo](https://gpudojo.com/a5000)
- RTX 5090: [bestvaluegpu](https://bestvaluegpu.com/en-eu/history/new-and-used-rtx-5090-price-history-and-specs/), [neoteo](https://www.neoteo.com/en/rtx-5090-prices-hit-eur4600-in-europe)
- RTX PRO 4000 Blackwell: [Newegg](https://www.newegg.com/nvidia-blackwell-900-5g147-2270-000-rtx-pro-4000-24gb-graphics-card/p/N82E16814132109), [Microless](https://global.microless.com/product/pny-nvidia-rtx-pro-4000-blackwell-graphics-card-24gb-gddr7-192-bit-memory-8960-cuda-cores-672-gb-s-memory-bandwidth-pci-express-5-0-x16-black-vcnrtxpro4000-sb/)
- RTX PRO 6000 Blackwell: [Tom's Hardware](https://www.tomshardware.com/pc-components/gpus/nvidia-doubles-rtx-pro-6000-blackwells-msrp-to-a-staggering-usd16-000-96gb-card-started-pre-orders-below-usd8-000-last-year)

**RAM prices:** [TechRadar](https://www.techradar.com/pro/good-news-ram-prices-seem-to-have-finally-stabilized-bad-news-its-probably-because-memory-prices-are-so-high-that-its-forcing-most-of-us-to-give-up-buying-anything), [GameGPU](https://en.gamegpu.com/news/zhelezo/tseny-na-ddr5-vzleteli-za-18-mesyatsev-na-500-i-prevysili-usd1250-za-64-gb)
