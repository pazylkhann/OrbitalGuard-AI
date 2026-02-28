# OrbitalGuard AI — Design System Specification

## Mission Control Dashboard v2.0

> **Figma-Style Design Concept**  
> Production-ready SaaS aesthetic with cosmic mission control theme

---

## 1. Design Philosophy

The OrbitalGuard AI dashboard embodies the **"Mission Control Center"** aesthetic — a professional, data-intensive interface that feels like a real aerospace operations dashboard. The design balances **information density** with **visual clarity**, ensuring operators can quickly assess satellite conjunction risks while being impressed by the visual presentation.

### Core Principles

1. **Depth through Glassmorphism** — Multi-layered transparency creates spatial hierarchy
2. **Motion as Information** — Animations convey system state and data changes
3. **Neon Accents on Void** — High-contrast neon colors against deep space backgrounds
4. **Professional Trust** — Clean typography and precise data presentation builds confidence

---

## 2. Color Palette

### Background Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│ --bg-void       #020408   Deepest space (body background)  │
│ --bg-deep       #050c14   Secondary background              │
│ --bg-space      #080f1c   Tertiary/section backgrounds    │
│ --bg-panel      #0b1525   Panel/card backgrounds            │
│ --bg-card       #0e1c30   Card surfaces                    │
│ --bg-elevated   #122038   Hover states, elevated elements  │
└─────────────────────────────────────────────────────────────┘
```

### Glass Effects

```
┌─────────────────────────────────────────────────────────────┐
│ --glass-bg       rgba(14, 28, 48, 0.72)  Primary glass      │
│ --glass-border   rgba(56, 139, 253, 0.18) Border glow      │
│ --glass-hover    rgba(56, 139, 253, 0.28) Hover state     │
│ --glass-blur     backdrop-filter: blur(12px)                │
└─────────────────────────────────────────────────────────────┘
```

### Primary Accent — Electric Cyan

```
┌─────────────────────────────────────────────────────────────┐
│ --accent-primary   #38bdf8   Main accent (buttons, links)  │
│ --accent-glow      #0ea5e9   Hover/active states           │
│ --accent-deep      #0369a1   Pressed/secondary             │
│ --accent-gradient  linear-gradient(135deg, #0ea5e9, #0369a1)│
└─────────────────────────────────────────────────────────────┘
```

### Neon Spectrum

| Token | Hex | Usage |
|-------|-----|-------|
| `--neon-cyan` | `#22d3ee` | Safe states, success, LEO orbit |
| `--neon-blue` | `#60a5fa` | Info, MEO orbit |
| `--neon-purple` | `#a78bfa` | Highlights, GEO orbit |
| `--neon-green` | `#34d399` | Positive metrics, low risk |
| `--neon-yellow` | `#fbbf24` | Warnings, maneuver windows |
| `--neon-orange` | `#fb923c` | Medium risk, attention |
| `--neon-red` | `#f87171` | Errors, critical alerts |

### Risk Assessment Colors

```
CRITICAL  ──── #ef4444  (--risk-critical)
HIGH      ──── #f97316  (--risk-high)
MEDIUM    ──── #eab308  (--risk-medium)
LOW       ──── #22c55e  (--risk-low)
```

### Text Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│ --text-primary   #e2e8f0   Headings, primary content       │
│ --text-secondary #94a3b8   Body text, descriptions          │
│ --text-muted    #475569   Labels, timestamps, metadata    │
│ --text-accent   #38bdf8   Links, highlighted values       │
└─────────────────────────────────────────────────────────────┘
```

### Border System

```
┌─────────────────────────────────────────────────────────────┐
│ --border-subtle  rgba(56, 139, 253, 0.10)  Subtle dividers│
│ --border-default rgba(56, 139, 253, 0.20)  Default borders │
│ --border-strong  rgba(56, 139, 253, 0.40)  Emphasized      │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Typography

### Font Stack

```css
--font-sans: 'Inter', system-ui, -apple-system, sans-serif;
--font-mono: 'JetBrains Mono', 'Fira Code', monospace;
```

### Type Scale

| Element | Size | Weight | Line Height | Letter Spacing |
|---------|------|--------|-------------|-----------------|
| Logo Brand | 15px | 800 | 1 | 0.08em |
| H1 | 28px | 800 | 1.2 | -0.02em |
| H2 | 22px | 700 | 1.3 | -0.01em |
| H3 | 18px | 600 | 1.4 | 0 |
| Body | 14px | 400 | 1.6 | 0 |
| Body Small | 13px | 400 | 1.5 | 0 |
| Caption | 12px | 500 | 1.4 | 0.02em |
| Label | 11px | 600 | 1.3 | 0.10em |
| Mono Data | 13px | 500 | 1.4 | 0.02em |

### Font Weights

- **300** — Light (rarely used)
- **400** — Regular (body text)
- **500** — Medium (emphasized body)
- **600** — Semibold (subheadings)
- **700** — Bold (headings)
- **800** — Extra Bold (brand, KPI values)
- **900** — Black (rarely used)

---

## 4. Spacing System

### Base Unit: 4px

```
┌─────────────────────────────────────────────────────────────┐
│ --space-xs   4px    Tight spacing, icon gaps               │
│ --space-sm   8px    Compact elements                       │
│ --space-md   16px   Standard spacing                       │
│ --space-lg   24px   Section spacing                        │
│ --space-xl   32px   Large gaps                             │
│ --space-2xl  48px   Major sections                         │
│ --space-3xl  64px   Page-level spacing                     │
└─────────────────────────────────────────────────────────────┘
```

### Usage Guidelines

- **Component internal padding**: `--space-md` (16px)
- **Card padding**: `--space-lg` (24px)
- **Section gaps**: `--space-xl` (32px)
- **Grid gap**: `--space-md` to `--space-lg`

---

## 5. Grid & Layout

### Max Content Width

- **Dashboard**: 1440px centered
- **Panels**: Fluid (100% of container)
- **Sidebar**: 380px fixed

### Dashboard Grid

```
┌─────────────────────────────────────────────────────────────┐
│                        TOPBAR (64px)                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────┐  ┌──────────────┐ │
│  │           KPI STRIP (5 cols)         │  │              │ │
│  └─────────────────────────────────────┘  │              │ │
│                                           │   SIDEBAR    │ │
│  ┌──────────────────┐  ┌──────────────┐  │  (Risk Gauge│ │
│  │   ORBIT CANVAS  │  │  RISK GAUGE  │  │   + Alerts)  │ │
│  │   (Visualization)│  │              │  │              │ │
│  └──────────────────┘  └──────────────┘  │              │ │
│                                           │              │ │
│  ┌─────────────────────────────────────┐  │              │ │
│  │         TIMELINE CHART              │  └──────────────┘ │
│  └─────────────────────────────────────┘                    │
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              CONJUNCTION EVENT CARDS                    ││
│  └─────────────────────────────────────────────────────────┘│
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                          FOOTER                             │
└─────────────────────────────────────────────────────────────┘
```

### Responsive Breakpoints

```css
/* Desktop Large */
@media (min-width: 1440px) {
  .dashboard-grid { grid-template-columns: 1fr 380px; }
}

/* Desktop */
@media (max-width: 1200px) {
  .dashboard-grid { grid-template-columns: 1fr; }
  .kpi-strip { grid-template-columns: repeat(3, 1fr); }
}

/* Tablet */
@media (max-width: 768px) {
  .topbar { padding: 0 var(--space-md); }
  .main-content { padding: var(--space-md); }
  .kpi-strip { grid-template-columns: repeat(2, 1fr); }
  .metrics-row { grid-template-columns: repeat(2, 1fr); }
  .nav-tabs { display: none; }
}

/* Mobile */
@media (max-width: 480px) {
  .kpi-strip { grid-template-columns: 1fr; }
  .panel-header { flex-direction: column; gap: var(--space-sm); }
}
```

---

## 6. Component Library

### 6.1 Cards (Glass Panels)

```css
.panel {
  background: var(--glass-bg);
  backdrop-filter: blur(12px);
  border: 1px solid var(--glass-border);
  border-radius: var(--radius-xl);
  box-shadow: var(--shadow-card);
  overflow: hidden;
}

.panel:hover {
  border-color: var(--glass-hover);
  box-shadow: var(--shadow-glow-blue);
}
```

**States:**
- Default: Subtle border, shadow
- Hover: Glowing border, elevated shadow
- Active: Pressed effect (translateY + darker shadow)

### 6.2 Buttons

**Primary Button:**
```css
.btn-primary {
  background: linear-gradient(135deg, var(--accent-glow), var(--accent-deep));
  color: white;
  border-radius: var(--radius-md);
  padding: 10px 20px;
  font-weight: 600;
  box-shadow: 0 0 16px rgba(14, 165, 233, 0.25);
  transition: all 0.2s ease;
}

.btn-primary:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 24px rgba(14, 165, 233, 0.4);
}

.btn-primary:active {
  transform: translateY(0);
}
```

**Secondary Button:**
```css
.btn-secondary {
  background: transparent;
  border: 1px solid var(--border-default);
  color: var(--text-secondary);
  border-radius: var(--radius-md);
}

.btn-secondary:hover {
  background: rgba(56, 189, 248, 0.08);
  border-color: var(--accent-primary);
  color: var(--accent-primary);
}
```

### 6.3 KPI Cards

```css
.kpi-card {
  background: var(--glass-bg);
  border: 1px solid var(--glass-border);
  border-radius: var(--radius-lg);
  padding: var(--space-lg);
  position: relative;
  overflow: hidden;
}

.kpi-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 2px;
  background: var(--kpi-accent, var(--accent-primary));
  opacity: 0.7;
}

.kpi-card::after {
  content: '';
  position: absolute;
  top: -40px;
  right: -20px;
  width: 80px;
  height: 80px;
  background: radial-gradient(circle, var(--kpi-accent) 0%, transparent 70%);
  opacity: 0.06;
}

.kpi-value {
  font-size: 32px;
  font-weight: 800;
  font-variant-numeric: tabular-nums;
}
```

### 6.4 Alert Badges

```css
.alert-badge {
  padding: 2px 8px;
  border-radius: var(--radius-full);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.badge-CRITICAL {
  background: rgba(239, 68, 68, 0.15);
  color: var(--risk-critical);
  border: 1px solid rgba(239, 68, 68, 0.30);
}

.badge-HIGH {
  background: rgba(249, 115, 22, 0.15);
  color: var(--risk-high);
  border: 1px solid rgba(249, 115, 22, 0.30);
}

.badge-MEDIUM {
  background: rgba(234, 179, 8, 0.15);
  color: var(--risk-medium);
  border: 1px solid rgba(234, 179, 8, 0.30);
}

.badge-LOW {
  background: rgba(34, 197, 94, 0.15);
  color: var(--risk-low);
  border: 1px solid rgba(34, 197, 94, 0.30);
}
```

### 6.5 Input Fields

```css
.input-field {
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  padding: 10px 14px;
  color: var(--text-primary);
  font-size: 14px;
  transition: all 0.2s ease;
}

.input-field:focus {
  outline: none;
  border-color: var(--accent-primary);
  box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.1);
}

.input-field::placeholder {
  color: var(--text-muted);
}
```

### 6.6 Navigation Tabs

```css
.nav-tab {
  padding: 6px 14px;
  border-radius: var(--radius-sm);
  font-size: 13px;
  font-weight: 500;
  color: var(--text-muted);
  background: none;
  border: 1px solid transparent;
  cursor: pointer;
  transition: all 0.2s ease;
}

.nav-tab:hover {
  color: var(--text-secondary);
  background: rgba(56, 189, 248, 0.06);
}

.nav-tab.active {
  color: var(--accent-primary);
  background: rgba(56, 189, 248, 0.10);
  border-color: var(--border-default);
}
```

---

## 7. Visual Effects

### 7.1 Glassmorphism

```css
.glass-panel {
  background: rgba(14, 28, 48, 0.72);
  backdrop-filter: blur(12px) saturate(180%);
  -webkit-backdrop-filter: blur(12px) saturate(180%);
  border: 1px solid rgba(56, 139, 253, 0.18);
  border-radius: var(--radius-xl);
}
```

### 7.2 Glow Effects

```css
/* Blue glow - default accent */
.glow-blue {
  box-shadow