"""
DocDeclutter — Transit Document Dataset Generator
Generates realistic, unstructured and semi-structured metro operational documents
aligned with Kochi Metro Rail Limited (KMRL) transit engineering domains (SIH25080).
"""

import os
import json

DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "documents")
EVAL_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "evaluation_set.json")

DATASET_DOCUMENTS = [
    {
        "doc_id": "KMRL-SOP-SIG-001",
        "title": "Standard Operating Procedure: CBTC Signaling & Train Control Subsystem",
        "doc_type": "Technical Manual / SOP",
        "publisher": "Kochi Metro Rail Limited - Signaling & Telecom Department",
        "date": "2024-01-15",
        "content": """================================================================================
KOCHI METRO RAIL LIMITED (KMRL) - SIGNALING & TELECOM DIVISION
DOCUMENT CODE: KMRL-SOP-SIG-001
REVISION: 3.2 | DATE: 15-JAN-2024
TITLE: STANDARD OPERATING PROCEDURE FOR CBTC SIGNALING & AUTOMATIC TRAIN SUPERVISION (ATS)
================================================================================

SECTION 1.0: SYSTEM OVERVIEW AND ARCHITECTURE
Kochi Metro operates an advanced Communications-Based Train Control (CBTC) signaling system over its 28-kilometer revenue alignment connecting Aluva to Thripunithura. The system utilizes redundant bidirectional radio communication links operating in the 5.8 GHz ISM frequency band between trackside Access Points (APs) and trainborne Vehicle On-Board Controllers (VOBC).

SECTION 2.0: AUTOMATIC TRAIN SUPERVISION (ATS) SUBSYSTEM
2.1 Automatic Train Supervision (ATS) at the Operation Control Centre (OCC) in Muttom monitors train movements in real time.
2.2 The primary ATS server handles automatic train routing, timetable regulation, headway management, and platform dwell time optimization.
2.3 Dwell times are dynamically adjusted between 20 seconds and 45 seconds depending on peak passenger interchange demand at interchange nodes like Edapally and MG Road.

SECTION 3.0: TRACKSIDE BALISE AND INTERLOCKING INTERFACE
3.1 Fixed Eurobalises are installed at 50-meter intervals along the main line to provide absolute position recalibration to the train's tachometer and Doppler radar sensors.
3.2 Electronic Interlocking (EI) interfaces with turnout point machines (type P-102). In normal operation, point throwing time must not exceed 4.5 seconds.
3.3 If point detection is lost during throwing, the EI will initiate two automatic re-trials before declaring a point failure alarm (Alarm Code: ALM-EI-PT-FAIL) to the OCC workstation.

SECTION 4.0: DEGRADED AND FALLBACK MODES
4.1 In the event of dual-radio communication timeout exceeding 3.5 seconds, the train VOBC immediately commands an Automatic Emergency Brake (EB) application, coming to a standstill within the target safety margin.
4.2 Transition to Restricted Manual (RM) mode requires formal authorization from the Traffic Controller (OCC). In RM mode, maximum train speed is rigidly capped at 25 km/h with cab signaling suppressed.
4.3 Reverse movements in mainline passenger areas are strictly prohibited unless the Track Detection Section is confirmed clear by axle counters.

SECTION 5.0: MAINTENANCE SCHEDULE AND FIELD INSPECTION
- Bi-weekly inspection: Visual inspection of wayside balise mountings, radio antenna alignments, and trackside cabling conduits.
- Monthly overhaul: Measurement of balise telegram signal strength (minimum acceptable threshold: -75 dBm) and verification of redundant antenna VSWR (< 1.5).
- Quarterly test: Complete ATS server failover drill and simulation of point machine detection failure.
"""
    },
    {
        "doc_id": "KMRL-MNT-PWR-002",
        "title": "Traction Power Substation (TSS) Maintenance Manual: 750V DC Third Rail System",
        "doc_type": "Maintenance Manual",
        "publisher": "Kochi Metro Rail Limited - Electrical & Traction Engineering",
        "date": "2024-02-10",
        "content": """================================================================================
KOCHI METRO RAIL LIMITED (KMRL) - TRACTION POWER & SUBSTATION
DOCUMENT CODE: KMRL-MNT-PWR-002
REVISION: 4.0 | DATE: 10-FEB-2024
TITLE: TRACTION POWER SUBSTATION (TSS) AND 750V DC THIRD RAIL MAINTENANCE MANUAL
================================================================================

SECTION 1.0: POWER DISTRIBUTION ARCHITECTURE
1.1 KMRL receives bulk electric power at 110 kV AC from the Kerala State Electricity Board (KSEB) grid at two primary Receiving Substations (RSS): Kaloor RSS and Muttom RSS.
1.2 Traction Substations (TSS) step down voltage through 33 kV traction transformers and convert it to 750V DC nominal voltage using 12-pulse diode rectifier units.
1.3 Traction current is delivered to 3-car EMU trainsets via a bottom-contact composite aluminium-steel third rail conductor.

SECTION 2.0: VACUUM CIRCUIT BREAKER (VCB-02) SPECIFICATIONS & TRIPPING PROCEDURES
2.1 Each 33 kV incoming feeder is protected by an indoor type Vacuum Circuit Breaker, designated model VCB-02.
2.2 Rated current capacity: 1250 A continuous; symmetrical breaking capacity: 25 kA at 33 kV; operating mechanism: motor-charged spring energy storage.
2.3 Causes of VCB-02 Tripping:
    - Overcurrent protection relay (Relay 51) pickup: Trip setting calibrated at 1400 A with a 0.2-second definite time delay.
    - Earth fault protection relay (Relay 51N) pickup: Trip setting at 120 A with instantaneous tripping (< 30 ms).
    - Transformer Buchholz gas accumulation alarm or pressure relief valve actuation.
2.4 Reset Procedure: Following an automatic VCB-02 lockout, maintenance technicians must inspect the breaker gas pressure gauge (nominal: 0.15 MPa), verify insulation resistance (> 500 Megohms using a 2.5 kV Megger), and obtain OCC traction clearance before manual closure.

SECTION 3.0: THIRD RAIL CONDUCTOR INSPECTION AND WEAR LIMITS
3.1 The stainless steel capping wear thickness is measured using calibrated ultrasonic thickness gauges during night engineering possessions (01:00 hrs to 04:30 hrs).
3.2 Initial cap thickness: 4.5 mm. Maximum allowable wear limit: 2.0 mm remaining. When thickness drops below 2.5 mm, replacement must be scheduled within 30 days.
3.3 Insulator bracket resistance must exceed 100 Megohms when measured in dry ambient conditions. Any bracket showing carbon tracking or hairline thermal cracking must be replaced immediately.

SECTION 4.0: STRAY CURRENT MITIGATION SYSTEM
4.1 To prevent electrochemical corrosion of civil viaduct rebars, a dedicated stray current collector cable is laid parallel to the running rails.
4.2 Stray current monitoring sensors are positioned at Aluva, Kalamassery, Kaloor, and Maharajas stations.
4.3 Track-to-earth potential must remain within the -5 V to +10 V window in accordance with EN 50122-2 standard.
"""
    },
    {
        "doc_id": "KMRL-RS-TM-003",
        "title": "Rolling Stock Maintenance Protocol: Traction Motor TM-104 Inspection and Overhaul",
        "doc_type": "Technical Instruction",
        "publisher": "Kochi Metro Rail Limited - Rolling Stock Division (Muttom Depot)",
        "date": "2024-03-22",
        "content": """================================================================================
KOCHI METRO RAIL LIMITED (KMRL) - ROLLING STOCK DIVISION
DOCUMENT CODE: KMRL-RS-TM-003
REVISION: 2.1 | DATE: 22-MAR-2024
TITLE: ROLLING STOCK TRACTION MOTOR TM-104 OVERHAUL AND BEARING TEMPERATURE PROTOCOL
================================================================================

SECTION 1.0: EQUIPMENT IDENTIFICATION AND RATINGS
1.1 Trainset Model: Alstom Metropolis 3-car trainset (Driving Motor Car DMA - Trailer Car TC - Driving Motor Car DMB).
1.2 Traction Motor Model: Three-phase squirrel-cage asynchronous motor designated TM-104.
1.3 Continuous rated power: 210 kW; Nominal line voltage: 550V AC three-phase supplied by VVVF inverter; Rated speed: 3200 RPM; Total motors per train: 8 units (4 per motor car).

SECTION 2.0: BEARING TEMPERATURE MONITORING AND THRESHOLDS
2.1 Each TM-104 motor is equipped with dual PT100 Resistance Temperature Detectors (RTD) embedded in drive-end (DE) and non-drive-end (NDE) bearing housings.
2.2 Operational Temperature Thresholds:
    - Normal operating band: 45°C to 75°C.
    - Caution advisory threshold: 85°C (triggers yellow telemetry banner on Train Management System - TMS).
    - Alarm threshold: 95°C (triggers acoustic warning chime in operator cab and reduces motor traction torque by 50%).
    - Critical Trip threshold: 105°C (TMS automatically isolates the affected VVVF traction inverter; train continues service on remaining motor bogies).
2.3 Bearing Lubrication: Kluberplex BEM 42-133 high-temperature synthetic grease. Greasing interval: Every 60,000 km or 6 months, injecting precisely 45 grams per bearing.

SECTION 3.0: STATOR INSULATION TESTING PROCEDURE
3.1 Phase-to-phase and phase-to-ground insulation resistance must be checked at every Scheduled Maintenance Interval (SMI-2, 30,000 km).
3.2 Test instrument: 1000V DC Insulation Tester. Acceptable baseline: Minimum 100 Megohms at 25°C ambient temperature.
3.3 If resistance is between 10 Megohms and 50 Megohms, perform thermal dehydration in the depot baking oven at 90°C for 6 hours. If below 10 Megohms, quarantine motor for stator rewinding.

SECTION 4.0: BOGIE WHEEL TREAD DIAMETER LIMITS
4.1 New wheel tread diameter: 860 mm. Condemning wheel limit: 780 mm.
4.2 Maximum permissible diameter discrepancy on same axle: 1.0 mm; on same bogie: 2.0 mm; across different bogies on same vehicle: 5.0 mm.
"""
    },
    {
        "doc_id": "KMRL-INC-2024-089",
        "title": "Incident Investigation Report: Service Delay at Aluva Terminal (14-Oct-2024)",
        "doc_type": "Incident Investigation Report",
        "publisher": "KMRL Safety & Operations Review Committee",
        "date": "2024-10-16",
        "content": """================================================================================
KOCHI METRO RAIL LIMITED (KMRL) - SAFETY INVESTIGATION REPORT
INCIDENT REFERENCE: KMRL-INC-2024-089
DATE OF OCCURRENCE: 14-OCT-2024 | TIME: 08:24 IST
LOCATION: ALUVA STATION - CROSSOVER TURNOUT 21A & PLATFORM 1
INVESTIGATING OFFICER: S. NAIR (CHIEF SAFETY COMMISSIONER)
================================================================================

SECTION 1.0: EXECUTIVE SUMMARY
On Monday, 14 October 2024, at approximately 08:24 IST during the morning peak revenue period, Trainset TS-07 operating Train Run No. 104 experienced an unexpected service disruption while approaching Aluva Station Platform 1. The train came to an abrupt automatic standstill 80 meters outside the station platform boundary, resulting in a line blockage and subsequent 24-minute cumulative service delay affecting eight downstream passenger trains.

SECTION 2.0: SEQUENCE OF EVENTS
- 08:22:15 IST: Train TS-07 dispatched from Pulinchodu Station towards Aluva Terminal on Down Line at 62 km/h.
- 08:23:48 IST: Train approaches crossover turnout 21A. The onboard VOBC reports packet timeout with trackside Radio Access Point AP-ALV-03.
- 08:23:52 IST: Loss of CBTC radio signal exceeds 3.5-second fail-safe timeout window.
- 08:23:55 IST: Train onboard computer commands fail-safe Emergency Brake (EB). Train TS-07 stops safely at Chainage 27+840.
- 08:25:00 IST: Train Driver initiates contact with OCC Traffic Controller reporting uncommanded EB application.
- 08:31:20 IST: OCC commands Traffic Controller to attempt radio link re-acquisition. Signal fails to establish.
- 08:35:40 IST: OCC grants formal clearance to switch Train TS-07 into Restricted Manual (RM) mode at 15 km/h.
- 08:42:10 IST: Train TS-07 successfully docks at Aluva Platform 1; all 340 passengers detrained safely without injury.
- 08:48:00 IST: Normal CBTC service restored after resetting wayside radio controller switch.

SECTION 3.0: ROOT CAUSE ANALYSIS
3.1 The technical investigation revealed that trackside Radio Access Point AP-ALV-03 suffered an unexpected power supply trip caused by moisture ingress into its external junction box JB-21A following heavy unseasonal rainfall the prior night.
3.2 The auxiliary DC-DC power converter inside JB-21A failed, dropping the 48V DC feed to the AP-ALV-03 antenna transmitter.
3.3 Secondary contributing factor: Redundant Access Point AP-ALV-04 was undergoing scheduled firmware migration and was temporarily offline, leaving the Aluva crossover zone without hot-standby radio coverage.

SECTION 4.0: CORRECTIVE AND PREVENTIVE ACTIONS (CAPA)
1. Replace weather-proofing gaskets on all 148 outdoor junction boxes across viaduct sections with IP68-rated silicon seals within 14 days.
2. Update OCC standard operating instructions: Never schedule radio maintenance or firmware upgrades during revenue operating hours (06:00 to 22:30 hrs).
3. Introduce dual redundant auxiliary power supplies inside each trackside junction box.
"""
    },
    {
        "doc_id": "KMRL-PWAY-TRN-005",
        "title": "Permanent Way Maintenance Report: Turnout Point Machine P-102 at Muttom Yard",
        "doc_type": "Maintenance Record",
        "publisher": "KMRL Track & Civil Infrastructure",
        "date": "2024-04-18",
        "content": """================================================================================
KOCHI METRO RAIL LIMITED (KMRL) - PERMANENT WAY (P-WAY) DIVISION
DOCUMENT CODE: KMRL-PWAY-TRN-005
INSPECTION DATE: 18-APR-2024 | LOCATION: MUTTOM DEPOT YARD - TURNOUT 102
EQUIPMENT: ELECTRO-MECHANICAL POINT MACHINE MODEL P-102
================================================================================

SECTION 1.0: INSPECTION OBJECTIVE
Routine quarterly preventive inspection and switch rail geometry verification for Point Machine P-102 controlling the reception line leading into the Muttom overhaul workshop.

SECTION 2.0: MEASUREMENT FINDINGS AND CLEARANCES
2.1 Switch Rail Opening (Throw): Measured at 115 mm on normal side, 114 mm on reverse side (Standard tolerance: 115 ± 2 mm). Status: PASS.
2.2 Tongue Rail to Stock Rail Housing: Clearance verified using a 1.5 mm feeler gauge. Gap measured at 0.8 mm (Permissible limit: < 2.0 mm). Status: PASS.
2.3 Point Machine Operating Current: During throwing cycle, motor drawn current was 2.8 A at 110V DC (Allowable upper limit: 3.5 A). Status: PASS.
2.4 Lock Rod and Detector Slide Clearance: Lock slide notch gap measured at 2.2 mm against standard 2.0 mm to 3.0 mm specification.

SECTION 3.0: LUBRICATION AND WEAR OBSERVATIONS
3.1 Slide chairs were cleaned of dried debris and lubricated using environmentally benign graphite grease grade KMRL-GG-4.
3.2 Minor surface spalling of 0.4 mm depth observed on right-hand tongue rail tip over an 80 mm length. Ground smooth using portable rail profile grinder in accordance with Track Standard TS-08.
3.3 Insulated rail joint (IRJ) resistance between turnout rails measured at 1.8 kOhms against minimum threshold of 1.0 kOhm.
"""
    },
    {
        "doc_id": "KMRL-AFC-LOG-006",
        "title": "Automatic Fare Collection (AFC) Technical Log: Maharajas College Station Gate Array",
        "doc_type": "Maintenance Log",
        "publisher": "KMRL IT & Revenue Systems",
        "date": "2024-05-02",
        "content": """================================================================================
KOCHI METRO RAIL LIMITED (KMRL) - AFC & REVENUE SYSTEMS
DOCUMENT CODE: KMRL-AFC-LOG-006
DATE: 02-MAY-2024 | STATION: MAHARAJAS COLLEGE STATION (CONCOURSE CON-A)
EQUIPMENT: AUTOMATIC FARE COLLECTION (AFC) FLAP BARRIER GATE ARRAY G-03
================================================================================

SECTION 1.0: REPORTED DEFECT
Station Controller logged recurring error code ERR-AFC-G03-VAL: Gate G-03 rejecting contactless smart cards and Kochi1 Rupay cards intermittently, with flap barrier staying open for 15 seconds after passenger passage.

SECTION 2.0: DIAGNOSTIC TESTING & FINDINGS
2.1 Diagnostic scan revealed the onboard QR/NFC card validator module (Model CV-880) was experiencing optical beam misalignment across infrared passenger tracking sensor array IR-04 and IR-05.
2.2 Dust accumulation from ongoing concourse platform ceiling renovation had coated the optical emitter diodes, attenuating infrared beam intensity by 42%.
2.3 Network switch port SW-CON-02 was exhibiting 8% packet drops to Station Computer (SC) due to an oxidized RJ45 termination.

SECTION 3.0: REPAIR WORK UNDERTAKEN
3.1 Cleaned all 16 pairs of optical beam sensors using 99% isopropyl alcohol wipes. Beam alignment recalibrated using laser sighting tool.
3.2 Re-terminated CAT-6 patch cable to network port SW-CON-02; packet loss dropped to 0.00%.
3.3 Flap barrier mechanical actuator torque re-adjusted to 18 Nm; transit passage cycle time verified at 0.7 seconds per passenger.
3.4 Test batch of 50 smart cards and 25 Kochi1 cards scanned successfully without rejection. Gate returned to active revenue service at 14:15 IST.
"""
    },
    {
        "doc_id": "KMRL-OPS-LOG-007",
        "title": "Daily Central Operation Control Centre (OCC) Log: 14-October-2024",
        "doc_type": "Daily Operations Record",
        "publisher": "KMRL Operations Department (Muttom OCC)",
        "date": "2024-10-14",
        "content": """================================================================================
KOCHI METRO RAIL LIMITED (KMRL) - OPERATION CONTROL CENTRE (OCC)
DAILY OPERATIONS LOG SHEET
DATE: 14-OCT-2024 | SHIFT: COMPREHENSIVE (05:00 - 23:30 IST)
CHIEF CONTROLLER ON DUTY: K. RADHAKRISHNAN
================================================================================

SECTION 1.0: OPERATIONAL SUMMARY METRICS
- Total Scheduled Train Trips: 246 trips
- Total Trips Operated: 244 trips (2 trips cancelled due to Incident KMRL-INC-2024-089)
- On-Time Punctuality Index: 96.7% (Target: 99.0%)
- Total Daily Ridership: 88,412 passengers
- Active Trainsets in Revenue Service: 16 trainsets (TS-01 to TS-16)
- Reserve Standby Trainsets: 3 trainsets (TS-17, TS-18 at Muttom Depot, TS-19 at Aluva Siding)
- Total Traction Energy Consumed: 48,210 kWh
- Regenerative Braking Energy Recovered: 14,860 kWh (30.8% energy recovery efficiency)

SECTION 2.0: NOTABLE LOGGED INCIDENTS
- 06:12 IST: Kalamassery Station Down Line Platform Screen Door (PSD) Gate 4 failed to open. Station staff placed manual door lock; trains operated normally. Rectified at 07:10 IST.
- 08:24 IST: INCIDENT KMRL-INC-2024-089. TS-07 uncommanded emergency brake near Aluva crossover 21A. 24-minute line delay. Managed via RM mode.
- 13:45 IST: Heavy monsoon downpour across Edapally - Palarivattom viaduct. Speed restriction of 50 km/h applied between Ch. 14+200 and 16+800 for 45 minutes due to visibility limits.
- 19:15 IST: Customer medical emergency at MG Road Station; passenger escorted by station first-aid team. Train TS-11 delayed by 2.5 minutes.
- 23:15 IST: Last revenue train docked at Aluva; third rail traction power de-energized at 23:45 IST for track maintenance possession.
"""
    },
    {
        "doc_id": "KMRL-SAF-EMG-008",
        "title": "Metro Safety Protocol: Mainline Viaduct Emergency Evacuation and Detrainment",
        "doc_type": "Safety Manual",
        "publisher": "KMRL Safety & Quality Assurance",
        "date": "2023-11-05",
        "content": """================================================================================
KOCHI METRO RAIL LIMITED (KMRL) - SAFETY DIRECTIVE
DOCUMENT CODE: KMRL-SAF-EMG-008
TITLE: EMERGENCY EVACUATION PROCEDURES FOR TRAINS STRANDED ON ELEVATED VIADUCT
================================================================================

SECTION 1.0: EVACUATION CRITERIA & OCC AUTHORIZATION
1.1 Detrainment of passengers onto the elevated viaduct is a LAST-RESORT safety measure.
1.2 Viaduct evacuation must be ordered ONLY when:
    - Train is disabled and cannot be moved under own power or rescue propulsion within 30 minutes.
    - Imminent life-safety hazard exists inside cars (e.g., uncontrolled onboard fire, toxic smoke ingress, structural damage).
1.3 NO EVACUATION MAY BEGIN WITHOUT EXPLICIT "TRACTION CURRENT DISCHARGE CONFIRMATION" (TCD) ISSUED BY THE OCC POWER CONTROLLER.

SECTION 2.0: TRACTION CURRENT DISCHARGE (TCD) MANDATE
2.1 The OCC Power Controller must remotely trip the corresponding 750V DC section circuit breakers.
2.2 The Station Supervisor or Train Driver must apply the portable trackside short-circuiting earthing device (red grounding clamp) across the third rail and running rail before opening train emergency doors.
2.3 The OCC Power Controller logs the unique TCD authorization token number in the emergency log book.

SECTION 3.0: PASSENGER DETRAINMENT PROCEDURE
3.1 The Train Operator deploys the front or rear end detrainment ramp located behind the driver cab bulkhead.
3.2 The maximum slope of the deployed telescopic ramp is 28 degrees; non-skid tread surfaces must be inspected before boarding passengers down.
3.3 Passengers are guided in single file along the dedicated 800 mm wide emergency evacuation walkway on the viaduct parapet wall toward the nearest station platform.
3.4 Walkway emergency guide lighting illuminates automatically (minimum 15 lux level maintained for 90 minutes via central UPS batteries).
"""
    },
    {
        "doc_id": "KMRL-TEL-TET-009",
        "title": "Telecom Engineering Specifications: TETRA Radio Network and Trackside Coverage",
        "doc_type": "Technical Specification",
        "publisher": "KMRL Telecom & SCADA Division",
        "date": "2024-06-11",
        "content": """================================================================================
KOCHI METRO RAIL LIMITED (KMRL) - TELECOMMUNICATIONS SPECIFICATION
DOCUMENT CODE: KMRL-TEL-TET-009
TITLE: TERRESTRIAL TRUNKED RADIO (TETRA) NETWORK OPERATION AND DISPATCH CONSOLES
================================================================================

SECTION 1.0: RADIO FREQUENCY ALLOCATION AND NETWORK TOPOLOGY
1.1 Voice and critical operational dispatch communication across KMRL employs a digital TETRA network operating in the dedicated government-assigned 380 - 400 MHz band.
1.2 Channel spacing: 25 kHz per carrier, with 4 time-division multiple access (TDMA) time slots per RF channel.
1.3 Base Station System (BSS) units are distributed at five strategic nodal stations: Aluva (BSS-01), Kalamassery (BSS-02), Kaloor (BSS-03), Ernakulam South (BSS-04), and Petta (BSS-05).

SECTION 2.0: TUNNEL AND UNDERGROUND SECTION COVERAGE
2.1 In underground tunnel segments and beneath elevated road flyovers where line-of-sight RF propagation is obstructed, radiating Leaky Coaxial Cable (LCX 7/8-inch) is suspended from tunnel ceiling brackets.
2.2 RF signal strength threshold: Minimum field strength of -85 dBm must be maintained across 99.5% of the track alignment with a bit error rate (BER) < 2%.
2.3 Emergency fallback: In the event of base station controller failure, handheld radios automatically switch to Direct Mode Operation (DMO) enabling peer-to-peer radio communication up to 1.5 km range.
"""
    },
    {
        "doc_id": "KMRL-MNT-HVC-010",
        "title": "Station Environmental Control: HVAC Chiller Unit CH-02 Trip Investigation at Vadakkekotta",
        "doc_type": "Maintenance Report",
        "publisher": "KMRL Electrical & Mechanical (E&M) Division",
        "date": "2024-07-19",
        "content": """================================================================================
KOCHI METRO RAIL LIMITED (KMRL) - E&M DEPARTMENT
DOCUMENT CODE: KMRL-MNT-HVC-010
DATE: 19-JUL-2024 | LOCATION: VADAKKEKOTTA STATION (UNDERGROUND PLANT ROOM)
EQUIPMENT: WATER-COOLED CENTRIFUGAL CHILLER UNIT CH-02 (CAPACITY: 350 TR)
================================================================================

SECTION 1.0: INCIDENT DESCRIPTION
At 11:35 IST on 19-July-2024, Chiller Unit CH-02 servicing the underground concourse and platform air conditioning system tripped on safety interlock ALM-HVC-DP-LOW, causing station concourse ambient temperature to rise from 24.5°C to 28.8°C over a 40-minute window.

SECTION 2.0: ROOT CAUSE FINDINGS
2.1 Inspection of the primary chilled water piping revealed the differential water pressure switch (DPS-02) across the shell-and-tube evaporator had sustained contact corrosion.
2.2 Although actual water flow was verified at 120 cubic meters/hour (well above the minimum required safety threshold of 95 m3/h), the corroded switch falsely signaled a low-flow condition to the Chiller Microprocessor Panel.
2.3 Condenser water pump CWP-02 had air trapped in its casing following suction strainer cleaning earlier that morning.

SECTION 3.0: REMEDIAL REPAIRS EXECUTED
3.1 Replaced faulty differential pressure switch DPS-02 with a hermetically sealed IP66 pressure transmitter (Model Danfoss RT-112).
3.2 De-aerated and bled air from pump casing CWP-02; balanced cooling tower return flow valves.
3.3 Restarted Chiller Unit CH-02 at 13:10 IST. Evaporator chilled water leaving temperature stabilized at 7.2°C, and concourse temperature normalized to 24.0°C within 35 minutes.
"""
    },
    {
        "doc_id": "KMRL-RS-BRK-011",
        "title": "Rolling Stock Pneumatic Brake System: Compressor ACU-01 and Brake Caliper Limits",
        "doc_type": "Maintenance Protocol",
        "publisher": "KMRL Rolling Stock Division",
        "date": "2024-08-04",
        "content": """================================================================================
KOCHI METRO RAIL LIMITED (KMRL) - ROLLING STOCK DIVISION
DOCUMENT CODE: KMRL-RS-BRK-011
TITLE: AIR COMPRESSOR UNIT ACU-01 MAINTENANCE AND PNEUMATIC BRAKE THRESHOLDS
================================================================================

SECTION 1.0: PNEUMATIC SYSTEM ARCHITECTURE
1.1 Compressed air for train mechanical braking, secondary air suspension bellows, and pantograph/third-rail shoe lifting is supplied by Air Compressor Unit model ACU-01.
1.2 ACU-01 is an oil-free reciprocating compressor mounted under the Trailer Car (TC) chassis.
1.3 Main Reservoir Pipe (MRP) nominal operating pressure is regulated between 8.5 bar (cut-in) and 10.0 bar (cut-out).

SECTION 2.0: EMERGENCY BRAKE APPLICATION THRESHOLDS
2.1 Emergency Brake (EB) is a fail-safe, spring-applied, pneumatically released friction brake system.
2.2 The brake pipe (BP) pressure is charged at 5.0 bar in running state. Any pressure drop below 3.2 bar triggers full EB application.
2.3 If MRP pressure drops below 6.2 bar, train traction is automatically inhibited by the Train Management System (TMS) to prevent departing a station with insufficient reservoir reserve.

SECTION 3.0: BRAKE PAD WEAR AND CALIPER INSPECTION
3.1 Sintered metallic brake pad initial thickness: 24.0 mm. Minimum condemning wear limit: 5.0 mm remaining backing plate clearance.
3.2 Brake cylinder piston stroke under full service brake must be 38 ± 4 mm. If stroke exceeds 45 mm, automatic slack adjuster must be inspected.
"""
    },
    {
        "doc_id": "KMRL-TRK-RL-012",
        "title": "Permanent Way Rail Ultrasonic Flaw Detection (USFD) and Rail Grinding Protocol",
        "doc_type": "Technical Guideline",
        "publisher": "KMRL Permanent Way Engineering",
        "date": "2024-09-08",
        "content": """================================================================================
KOCHI METRO RAIL LIMITED (KMRL) - PERMANENT WAY ENGINEERING
DOCUMENT CODE: KMRL-TRK-RL-012
TITLE: ULTRASONIC FLAW DETECTION (USFD) TESTING AND CORRUGATION GRINDING
================================================================================

SECTION 1.0: USFD TESTING FREQUENCY
1.1 Mainline track rail section: 60 kg/m UIC 60 steel rails with 1080 Head Hardened (HH) grade steel on sharp viaduct curves (radius < 400 m).
1.2 Ultrasonic Flaw Detection (USFD) testing of rails and welded joints is conducted every 4 months using a digital multi-channel rail tester (Model Sperry DRT-44).
1.3 Any flaw classified as 'Immediate Removal Defect' (Code IMR) requires clamping the rail section with emergency fishplates and imposing a 20 km/h speed limit within 2 hours.

SECTION 2.0: RAIL CORRUGATION AND ACOUSTIC ROUGHNESS
2.1 Short-pitch corrugation wavelength (30 mm to 80 mm) occurs on tight curves between Changampuzha Park and Palarivattom.
2.2 When acoustic rail roughness exceeds 15 micrometers (peak-to-valley), a specialized rail grinding train (RGT) is deployed during night possessions.
2.3 Grinding passes remove 0.2 mm to 0.4 mm of metal per pass to restore the nominal transverse rail profile and eliminate Rolling Contact Fatigue (RCF) microcracks.
"""
    }
]

EVALUATION_QUESTIONS = [
    {
        "query_id": "Q01",
        "category": "Signaling & CBTC",
        "question": "What is the maximum speed permitted when a metro train is operated in Restricted Manual (RM) mode?",
        "expected_doc_id": "KMRL-SOP-SIG-001",
        "expected_section": "SECTION 4.0: DEGRADED AND FALLBACK MODES",
        "expected_page": 1,
        "supporting_text": "In RM mode, maximum train speed is rigidly capped at 25 km/h with cab signaling suppressed.",
        "expected_answer": "In Restricted Manual (RM) mode, the maximum train speed is rigidly capped at 25 km/h, with cab signaling suppressed, and requires formal authorization from the Traffic Controller (OCC).",
        "is_answerable": True
    },
    {
        "query_id": "Q02",
        "category": "Signaling & CBTC",
        "question": "What happens if point detection is lost during throwing of the turnout point machine?",
        "expected_doc_id": "KMRL-SOP-SIG-001",
        "expected_section": "SECTION 3.0: TRACKSIDE BALISE AND INTERLOCKING INTERFACE",
        "expected_page": 1,
        "supporting_text": "If point detection is lost during throwing, the EI will initiate two automatic re-trials before declaring a point failure alarm (Alarm Code: ALM-EI-PT-FAIL) to the OCC workstation.",
        "expected_answer": "The Electronic Interlocking (EI) will initiate two automatic re-trials before declaring a point failure alarm (Alarm Code: ALM-EI-PT-FAIL) to the OCC workstation.",
        "is_answerable": True
    },
    {
        "query_id": "Q03",
        "category": "Signaling & CBTC",
        "question": "How long must dual-radio communication timeout occur before the train VOBC commands an Emergency Brake application?",
        "expected_doc_id": "KMRL-SOP-SIG-001",
        "expected_section": "SECTION 4.0: DEGRADED AND FALLBACK MODES",
        "expected_page": 1,
        "supporting_text": "In the event of dual-radio communication timeout exceeding 3.5 seconds, the train VOBC immediately commands an Automatic Emergency Brake (EB) application",
        "expected_answer": "A dual-radio communication timeout exceeding 3.5 seconds triggers the train VOBC to command an Automatic Emergency Brake (EB) application.",
        "is_answerable": True
    },
    {
        "query_id": "Q04",
        "category": "Traction Power",
        "question": "What are the pickup settings and delay times for the overcurrent and earth fault protection relays on the VCB-02 breaker?",
        "expected_doc_id": "KMRL-MNT-PWR-002",
        "expected_section": "SECTION 2.0: VACUUM CIRCUIT BREAKER (VCB-02) SPECIFICATIONS & TRIPPING PROCEDURES",
        "expected_page": 1,
        "supporting_text": "Overcurrent protection relay (Relay 51) pickup: Trip setting calibrated at 1400 A with a 0.2-second definite time delay. Earth fault protection relay (Relay 51N) pickup: Trip setting at 120 A with instantaneous tripping (< 30 ms).",
        "expected_answer": "The overcurrent relay (Relay 51) is calibrated at 1400 A with a 0.2-second definite time delay, while the earth fault relay (Relay 51N) is set at 120 A with instantaneous tripping (< 30 ms).",
        "is_answerable": True
    },
    {
        "query_id": "Q05",
        "category": "Traction Power",
        "question": "What is the maximum allowable wear limit for the third rail stainless steel capping?",
        "expected_doc_id": "KMRL-MNT-PWR-002",
        "expected_section": "SECTION 3.0: THIRD RAIL CONDUCTOR INSPECTION AND WEAR LIMITS",
        "expected_page": 1,
        "supporting_text": "Initial cap thickness: 4.5 mm. Maximum allowable wear limit: 2.0 mm remaining. When thickness drops below 2.5 mm, replacement must be scheduled within 30 days.",
        "expected_answer": "The initial cap thickness is 4.5 mm and the maximum allowable wear limit is 2.0 mm remaining thickness (with replacement scheduled within 30 days once it drops below 2.5 mm).",
        "is_answerable": True
    },
    {
        "query_id": "Q06",
        "category": "Rolling Stock",
        "question": "What are the temperature thresholds for TM-104 traction motor bearings, and at what temperature is the motor inverter isolated?",
        "expected_doc_id": "KMRL-RS-TM-003",
        "expected_section": "SECTION 2.0: BEARING TEMPERATURE MONITORING AND THRESHOLDS",
        "expected_page": 1,
        "supporting_text": "Normal operating band: 45°C to 75°C. Caution advisory threshold: 85°C... Alarm threshold: 95°C... Critical Trip threshold: 105°C (TMS automatically isolates the affected VVVF traction inverter)",
        "expected_answer": "Normal operating range is 45°C-75°C; Caution is 85°C; Alarm is 95°C; and at the Critical Trip threshold of 105°C, the TMS automatically isolates the affected VVVF traction inverter.",
        "is_answerable": True
    },
    {
        "query_id": "Q07",
        "category": "Rolling Stock",
        "question": "What type of synthetic grease is used for lubricating TM-104 traction motor bearings and what is the injection amount?",
        "expected_doc_id": "KMRL-RS-TM-003",
        "expected_section": "SECTION 2.0: BEARING TEMPERATURE MONITORING AND THRESHOLDS",
        "expected_page": 1,
        "supporting_text": "Bearing Lubrication: Kluberplex BEM 42-133 high-temperature synthetic grease. Greasing interval: Every 60,000 km or 6 months, injecting precisely 45 grams per bearing.",
        "expected_answer": "Kluberplex BEM 42-133 high-temperature synthetic grease is used, injecting precisely 45 grams per bearing every 60,000 km or 6 months.",
        "is_answerable": True
    },
    {
        "query_id": "Q08",
        "category": "Incident Investigation",
        "question": "What caused the metro service delay on 14 October 2024 at Aluva Station?",
        "expected_doc_id": "KMRL-INC-2024-089",
        "expected_section": "SECTION 3.0: ROOT CAUSE ANALYSIS",
        "expected_page": 1,
        "supporting_text": "trackside Radio Access Point AP-ALV-03 suffered an unexpected power supply trip caused by moisture ingress into its external junction box JB-21A following heavy unseasonal rainfall the prior night. The auxiliary DC-DC power converter inside JB-21A failed, dropping the 48V DC feed... Redundant Access Point AP-ALV-04 was undergoing scheduled firmware migration and was temporarily offline",
        "expected_answer": "The delay on 14 October 2024 at Aluva was caused by trackside Radio Access Point AP-ALV-03 losing power due to moisture ingress into junction box JB-21A after heavy rain, while redundant AP-ALV-04 was offline for firmware migration, causing a CBTC radio timeout and automatic emergency brake on Train TS-07.",
        "is_answerable": True
    },
    {
        "query_id": "Q09",
        "category": "Incident Investigation",
        "question": "What were the immediate corrective and preventive actions ordered following the Aluva service disruption?",
        "expected_doc_id": "KMRL-INC-2024-089",
        "expected_section": "SECTION 4.0: CORRECTIVE AND PREVENTIVE ACTIONS (CAPA)",
        "expected_page": 1,
        "supporting_text": "1. Replace weather-proofing gaskets on all 148 outdoor junction boxes across viaduct sections with IP68-rated silicon seals within 14 days. 2. Update OCC standard operating instructions: Never schedule radio maintenance or firmware upgrades during revenue operating hours... 3. Introduce dual redundant auxiliary power supplies",
        "expected_answer": "The corrective actions were: replacing weather-proofing gaskets on all 148 outdoor junction boxes with IP68 silicon seals within 14 days; prohibiting radio maintenance/firmware upgrades during revenue hours (06:00-22:30); and introducing dual redundant auxiliary power supplies inside trackside junction boxes.",
        "is_answerable": True
    },
    {
        "query_id": "Q10",
        "category": "Permanent Way & Track",
        "question": "What was the measured operating current of Point Machine P-102 during inspection at Muttom Yard?",
        "expected_doc_id": "KMRL-PWAY-TRN-005",
        "expected_section": "SECTION 2.0: MEASUREMENT FINDINGS AND CLEARANCES",
        "expected_page": 1,
        "supporting_text": "During throwing cycle, motor drawn current was 2.8 A at 110V DC (Allowable upper limit: 3.5 A). Status: PASS.",
        "expected_answer": "The motor drawn current for Point Machine P-102 was 2.8 A at 110V DC, which passed within the allowable upper limit of 3.5 A.",
        "is_answerable": True
    },
    {
        "query_id": "Q11",
        "category": "Fare Collection & AFC",
        "question": "Why was AFC gate G-03 rejecting smart cards at Maharajas College Station?",
        "expected_doc_id": "KMRL-AFC-LOG-006",
        "expected_section": "SECTION 2.0: DIAGNOSTIC TESTING & FINDINGS",
        "expected_page": 1,
        "supporting_text": "Dust accumulation from ongoing concourse platform ceiling renovation had coated the optical emitter diodes, attenuating infrared beam intensity by 42%. Network switch port SW-CON-02 was exhibiting 8% packet drops to Station Computer (SC) due to an oxidized RJ45 termination.",
        "expected_answer": "AFC Gate G-03 was failing because dust from concourse ceiling renovation coated the optical sensors attenuating infrared beam intensity by 42%, compounded by an oxidized RJ45 network port causing an 8% packet drop.",
        "is_answerable": True
    },
    {
        "query_id": "Q12",
        "category": "Daily Operations",
        "question": "What was the total daily ridership and on-time punctuality index for Kochi Metro on 14 October 2024?",
        "expected_doc_id": "KMRL-OPS-LOG-007",
        "expected_section": "SECTION 1.0: OPERATIONAL SUMMARY METRICS",
        "expected_page": 1,
        "supporting_text": "Total Daily Ridership: 88,412 passengers. On-Time Punctuality Index: 96.7% (Target: 99.0%)",
        "expected_answer": "On 14 October 2024, total daily ridership was 88,412 passengers and the on-time punctuality index was 96.7% (against a target of 99.0%).",
        "is_answerable": True
    },
    {
        "query_id": "Q13",
        "category": "Safety & Evacuation",
        "question": "What safety confirmation is strictly required from the OCC before passengers can be detrained onto an elevated viaduct?",
        "expected_doc_id": "KMRL-SAF-EMG-008",
        "expected_section": "SECTION 1.0: EVACUATION CRITERIA & OCC AUTHORIZATION",
        "expected_page": 1,
        "supporting_text": "NO EVACUATION MAY BEGIN WITHOUT EXPLICIT 'TRACTION CURRENT DISCHARGE CONFIRMATION' (TCD) ISSUED BY THE OCC POWER CONTROLLER.",
        "expected_answer": "No passenger evacuation can begin without explicit 'Traction Current Discharge Confirmation' (TCD) issued by the OCC Power Controller, followed by applying trackside short-circuit earthing clamps.",
        "is_answerable": True
    },
    {
        "query_id": "Q14",
        "category": "Telecommunications",
        "question": "What frequency band and channel spacing does the KMRL TETRA radio network use?",
        "expected_doc_id": "KMRL-TEL-TET-009",
        "expected_section": "SECTION 1.0: RADIO FREQUENCY ALLOCATION AND NETWORK TOPOLOGY",
        "expected_page": 1,
        "supporting_text": "dedicated government-assigned 380 - 400 MHz band. Channel spacing: 25 kHz per carrier, with 4 time-division multiple access (TDMA) time slots per RF channel.",
        "expected_answer": "KMRL's TETRA radio network operates in the 380 - 400 MHz band with 25 kHz channel spacing and 4 TDMA time slots per carrier.",
        "is_answerable": True
    },
    {
        "query_id": "Q15",
        "category": "HVAC & Station Environment",
        "question": "Why did Chiller Unit CH-02 trip at Vadakkekotta Station and what replacement part was installed?",
        "expected_doc_id": "KMRL-MNT-HVC-010",
        "expected_section": "SECTION 2.0: ROOT CAUSE FINDINGS & SECTION 3.0",
        "expected_page": 1,
        "supporting_text": "differential water pressure switch (DPS-02) across the shell-and-tube evaporator had sustained contact corrosion... falsely signaled a low-flow condition... Replaced faulty differential pressure switch DPS-02 with a hermetically sealed IP66 pressure transmitter (Model Danfoss RT-112).",
        "expected_answer": "Chiller CH-02 tripped because evaporator differential pressure switch DPS-02 had contact corrosion and falsely signaled low water flow; it was replaced with an IP66 Danfoss RT-112 pressure transmitter.",
        "is_answerable": True
    },
    {
        "query_id": "Q16",
        "category": "Rolling Stock & Pneumatics",
        "question": "At what Main Reservoir Pipe (MRP) pressure does the Train Management System inhibit train traction?",
        "expected_doc_id": "KMRL-RS-BRK-011",
        "expected_section": "SECTION 2.0: EMERGENCY BRAKE APPLICATION THRESHOLDS",
        "expected_page": 1,
        "supporting_text": "If MRP pressure drops below 6.2 bar, train traction is automatically inhibited by the Train Management System (TMS) to prevent departing a station with insufficient reservoir reserve.",
        "expected_answer": "Train traction is automatically inhibited if Main Reservoir Pipe (MRP) pressure drops below 6.2 bar.",
        "is_answerable": True
    },
    {
        "query_id": "Q17",
        "category": "Permanent Way & Track",
        "question": "How often is Ultrasonic Flaw Detection (USFD) testing conducted on KMRL rails and what action is required for an IMR defect?",
        "expected_doc_id": "KMRL-TRK-RL-012",
        "expected_section": "SECTION 1.0: USFD TESTING FREQUENCY",
        "expected_page": 1,
        "supporting_text": "testing of rails and welded joints is conducted every 4 months using a digital multi-channel rail tester (Model Sperry DRT-44). Any flaw classified as 'Immediate Removal Defect' (Code IMR) requires clamping the rail section with emergency fishplates and imposing a 20 km/h speed limit within 2 hours.",
        "expected_answer": "USFD testing is conducted every 4 months; an 'Immediate Removal Defect' (IMR) requires emergency fishplate clamping and a 20 km/h speed restriction within 2 hours.",
        "is_answerable": True
    },
    # Unanswerable queries (Abstention tests)
    {
        "query_id": "Q18",
        "category": "Unanswerable / Abstention",
        "question": "What is the ticket price for a journey from Delhi Metro Kashmere Gate to Noida City Centre?",
        "expected_doc_id": None,
        "expected_section": None,
        "expected_page": None,
        "supporting_text": None,
        "expected_answer": "The available documents do not provide enough evidence to answer this question.",
        "is_answerable": False
    },
    {
        "query_id": "Q19",
        "category": "Unanswerable / Abstention",
        "question": "Which catering vendor supplies meals on the Vande Bharat Express between Mumbai and Ahmedabad?",
        "expected_doc_id": None,
        "expected_section": None,
        "expected_page": None,
        "supporting_text": None,
        "expected_answer": "The available documents do not provide enough evidence to answer this question.",
        "is_answerable": False
    },
    {
        "query_id": "Q20",
        "category": "Unanswerable / Abstention",
        "question": "What is the solar panel battery warranty period for the London Underground Piccadilly line?",
        "expected_doc_id": None,
        "expected_section": None,
        "expected_page": None,
        "supporting_text": None,
        "expected_answer": "The available documents do not provide enough evidence to answer this question.",
        "is_answerable": False
    }
]


def generate_documents():
    """Write text and PDF documents to data/documents/"""
    os.makedirs(DOCS_DIR, exist_ok=True)
    
    # 1. Write text documents
    for doc in DATASET_DOCUMENTS:
        filename = f"{doc['doc_id']}.txt"
        filepath = os.path.join(DOCS_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(doc["content"])
        print(f"Generated text document: {filename}")

    # 2. Write PDF documents using PyMuPDF if available
    try:
        import pymupdf
        for doc in DATASET_DOCUMENTS:
            pdf_filename = f"{doc['doc_id']}.pdf"
            pdf_path = os.path.join(DOCS_DIR, pdf_filename)
            pdf_doc = pymupdf.open()
            
            lines = doc["content"].split("\n")
            page_text_blocks = []
            current_block = []
            
            # Divide into pages of roughly 35 lines
            for line in lines:
                current_block.append(line)
                if len(current_block) >= 35:
                    page_text_blocks.append("\n".join(current_block))
                    current_block = []
            if current_block:
                page_text_blocks.append("\n".join(current_block))
                
            for p_idx, text_block in enumerate(page_text_blocks):
                page = pdf_doc.new_page(width=595, height=842) # A4
                rect = pymupdf.Rect(50, 50, 545, 792)
                page.insert_textbox(rect, text_block, fontsize=9.5, fontname="helv", lineheight=1.2)
                
            pdf_doc.save(pdf_path)
            pdf_doc.close()
            print(f"Generated PDF document: {pdf_filename}")
    except Exception as e:
        print(f"Notice: PyMuPDF PDF generation skipped or error: {e}")

    # 3. Write evaluation dataset
    os.makedirs(os.path.dirname(EVAL_FILE), exist_ok=True)
    with open(EVAL_FILE, "w", encoding="utf-8") as f:
        json.dump(EVALUATION_QUESTIONS, f, indent=2)
    print(f"Generated evaluation dataset: {EVAL_FILE} ({len(EVALUATION_QUESTIONS)} test cases)")


if __name__ == "__main__":
    generate_documents()
