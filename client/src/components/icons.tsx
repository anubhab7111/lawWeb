// Shared line-icon set. One stroke weight, one visual language — so icons read
// as a family across the chat, directory, vault and payment screens rather than
// a grab-bag of emoji. All inherit `currentColor`.

const base = {
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.6,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
};

type IconProps = { size?: number };

const svgProps = (size: number) => ({ width: size, height: size, viewBox: "0 0 24 24", ...base });

export const IconSearch = ({ size = 15 }: IconProps) => (
  <svg {...svgProps(size)}><circle cx="11" cy="11" r="7" /><path d="m20 20-3.2-3.2" /></svg>
);
export const IconPlus = ({ size = 15 }: IconProps) => (
  <svg {...svgProps(size)}><path d="M12 5v14M5 12h14" /></svg>
);
export const IconPaperclip = ({ size = 16 }: IconProps) => (
  <svg {...svgProps(size)}><path d="M21 8.5 12.3 17.2a4.5 4.5 0 0 1-6.4-6.4l8.5-8.5a3 3 0 0 1 4.2 4.2l-8.5 8.5a1.5 1.5 0 0 1-2.1-2.1l7.8-7.8" /></svg>
);
export const IconClose = ({ size = 13 }: IconProps) => (
  <svg {...svgProps(size)}><path d="M6 6l12 12M18 6 6 18" /></svg>
);
export const IconBell = ({ size = 18 }: IconProps) => (
  <svg {...svgProps(size)}><path d="M18 8a6 6 0 0 0-12 0c0 7-3 9-3 9h18s-3-2-3-9" /><path d="M13.7 21a2 2 0 0 1-3.4 0" /></svg>
);
export const IconUpload = ({ size = 24 }: IconProps) => (
  <svg {...svgProps(size)}><path d="M12 15V3" /><path d="m7 8 5-5 5 5" /><path d="M4 15v4a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-4" /></svg>
);
export const IconLock = ({ size = 14 }: IconProps) => (
  <svg {...svgProps(size)}><rect x="4.5" y="10.5" width="15" height="10" rx="2" /><path d="M8 10.5V7a4 4 0 0 1 8 0v3.5" /></svg>
);
export const IconCheck = ({ size = 14 }: IconProps) => (
  <svg {...svgProps(size)}><path d="M4 12.5 9 17.5 20 6.5" /></svg>
);
export const IconArrowLeft = ({ size = 14 }: IconProps) => (
  <svg {...svgProps(size)}><path d="M20 12H4" /><path d="m10 6-6 6 6 6" /></svg>
);
export const IconScale = ({ size = 15 }: IconProps) => (
  <svg {...svgProps(size)}><path d="M12 3v18" /><path d="M7 21h10" /><path d="M5 7h14" /><path d="M8 6.5 5 13a3 3 0 0 0 6 0Z" /><path d="M16 6.5 13 13a3 3 0 0 0 6 0Z" /></svg>
);
export const IconCalendar = ({ size = 15 }: IconProps) => (
  <svg {...svgProps(size)}><rect x="3.5" y="5" width="17" height="15" rx="2" /><path d="M3.5 9.5h17" /><path d="M8 3.5v3M16 3.5v3" /></svg>
);
