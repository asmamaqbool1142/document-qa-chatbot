"""Generate the sample PDF manual used by the demo and the evaluation set.

The product and company are fictional. Run from the project root:
    python scripts/generate_sample_pdf.py
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "samples" / "aquapure_w300_manual.pdf"

PAGES = [
    (
        "AquaPure W300 Under-Sink Water Purifier - User Manual",
        """Thank you for choosing the AquaPure W300, manufactured by Clearwater Appliances Ltd. \
This manual covers installation, daily use, filter maintenance, troubleshooting and warranty.

1. Technical Specifications

Model number: W300-EU (230 V) and W300-US (120 V).
Purification stages: 4 (sediment pre-filter, activated carbon block, reverse osmosis (RO) \
membrane, mineral post-filter).
Maximum flow rate: 2.5 litres per minute.
Storage tank capacity: 8 litres.
Rated power consumption: 45 W during filtration; under 1 W in standby.
Operating water pressure: 1.0 to 4.0 bar. Install a pressure-reducing valve if mains \
pressure exceeds 4.0 bar.
Operating water temperature: 5 to 38 degrees Celsius.
Dimensions: 42 cm x 18 cm x 38 cm. Weight: 9.6 kg (empty).
Waste water ratio: 1 litre of concentrate for every 2 litres of purified water.""",
    ),
    (
        "2. Installation",
        """The AquaPure W300 is designed to be installed under a kitchen sink by an adult with \
basic plumbing experience. Installation normally takes about 45 minutes.

Before you start, close the cold-water supply valve and disconnect the purifier from \
power. Connect the feed adapter to the cold-water line only; never connect the unit to a \
hot-water supply, because water above 38 degrees Celsius permanently damages the RO membrane.

Mount the faucet on the sink top using the supplied 12 mm drill guide. Connect the blue \
tube to the faucet, the white tube to the cold-water feed and the red tube to the drain \
saddle. Leave at least 10 cm of free space above the unit for filter changes.

After installation, flush the system for 15 minutes and discard the first full tank \
(8 litres) of water. This removes carbon fines and preservative from the new membrane.""",
    ),
    (
        "3. Filter Replacement Schedule",
        """Replace the filters on the following schedule, or sooner if the filter indicator light \
turns red:

Stage 1 - Sediment pre-filter (part AP-S1): every 6 months.
Stage 2 - Activated carbon block (part AP-C2): every 12 months.
Stage 3 - Reverse osmosis membrane (part AP-RO3): every 24 months.
Stage 4 - Mineral post-filter (part AP-M4): every 12 months.

To replace a filter, turn off the water supply, press the pressure-release button for \
5 seconds, then twist the filter housing a quarter-turn anticlockwise. After fitting a new \
filter, hold the RESET button for 3 seconds until the indicator turns green.

Using non-genuine filters may reduce purification performance and voids the warranty on \
the RO membrane.""",
    ),
    (
        "4. Troubleshooting and Error Codes",
        """E1 - Low inlet pressure. Check that the supply valve is fully open. If pressure is \
below 1.0 bar, a booster pump (accessory AP-BP1) is required.
E2 - Leak detected. The unit stops automatically. Disconnect power, dry the leak sensor \
tray and check all tube connections before restarting.
E3 - Filter life expired. Replace the filter whose indicator is red, then reset.
E4 - Tank over-temperature. The water in the tank is above 38 degrees Celsius. Move the \
unit away from heat sources such as dishwashers.

If the purified water tastes salty, the RO membrane may be damaged and should be replaced. \
For any other problem, contact Clearwater support at the number on the product label.""",
    ),
    (
        "5. Warranty",
        """Clearwater Appliances Ltd warrants the AquaPure W300 against defects in materials and \
workmanship for 2 years from the date of purchase. The stainless-steel storage tank is \
covered for 5 years. Replacement filters are consumables and are not covered by the \
warranty.

The warranty does not cover damage caused by connecting the unit to hot water, by \
freezing, by installation that does not follow this manual, or by using non-genuine \
filters. To make a claim, keep your proof of purchase and register the product within \
30 days of purchase at the Clearwater website.

Registered owners can buy an extended warranty plan that adds 3 more years of cover for \
the electronic control board. The extended plan must be purchased within the first 12 \
months of ownership.""",
    ),
]


def build() -> Path:
    doc = pymupdf.open()
    for title, body in PAGES:
        page = doc.new_page(width=595, height=842)  # A4 in points
        page.insert_text((56, 72), title, fontsize=15, fontname="helv")
        rect = pymupdf.Rect(56, 92, 539, 800)
        overflow = page.insert_textbox(rect, body, fontsize=10.5, fontname="helv", lineheight=1.35)
        if overflow < 0:
            raise RuntimeError(f"Text overflowed the page: {title}")
        page.insert_text((56, 820), f"AquaPure W300 manual - page {page.number + 1}", fontsize=8)
    doc.set_metadata({"title": "AquaPure W300 User Manual (fictional sample)"})
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT, garbage=4, deflate=True)
    doc.close()
    return OUTPUT


if __name__ == "__main__":
    print(f"Wrote {build()}")
