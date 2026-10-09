// Files picked one by one, a picked folder, or a dropped folder: one list of files, each with the name it will be
// stored and cited under. A folder's subfolders become part of that name ("Unit 3/Part I.mp4" -> "Unit 3 - Part I.mp4"),
// so two files of the same name in different subfolders can't overwrite each other and stay recognisable.

export interface PickedFile {
  file: File;
  name: string;
}

export interface Picked {
  files: PickedFile[];
  /** Files left out because they are not of the kind chosen for the batch */
  skipped: number;
  /** The picked or dropped folder's own name, when there was one */
  folder: string | null;
}

/** "Lectures/Unit 3/Part I.mp4" -> "Unit 3 - Part I.mp4": the top folder is the batch itself, so it is dropped. */
export function uploadName(relativePath: string): string {
  const parts = relativePath.split(/[\\/]+/).filter(Boolean);
  const inner = parts.length > 1 ? parts.slice(1) : parts;
  return inner.join(" - ");
}

/** The same name twice becomes "name (2).ext", in order. */
export function dedupe(names: string[]): string[] {
  const seen = new Map<string, number>();
  return names.map((name) => {
    const key = name.toLowerCase();
    const count = (seen.get(key) ?? 0) + 1;
    seen.set(key, count);
    if (count === 1) return name;
    const dot = name.lastIndexOf(".");
    return dot > 0 ? `${name.slice(0, dot)} (${count})${name.slice(dot)}` : `${name} (${count})`;
  });
}

export function accepts(name: string, accept: string): boolean {
  const lower = name.toLowerCase();
  return accept.split(",").some((suffix) => lower.endsWith(suffix.trim()));
}

/** Files with their path inside a folder ("" for a file picked on its own), filtered and named. */
export function pick(entries: { file: File; path: string }[], accept: string): Picked {
  const kept = entries.filter((e) => !e.file.name.startsWith(".") && accepts(e.file.name, accept));
  const names = dedupe(kept.map((e) => (e.path ? uploadName(e.path) : e.file.name)));
  const folders = entries.map((e) => e.path.split(/[\\/]/)[0]).filter((p, i) => entries[i].path.includes("/") && p);
  return {
    files: kept.map((e, i) => ({ file: e.file, name: names[i] })),
    skipped: entries.length - kept.length,
    folder: folders.length ? folders[0] : null,
  };
}

/** From an <input> (a folder picker gives each file its webkitRelativePath). */
export function fromInput(list: FileList | null): { file: File; path: string }[] {
  return Array.from(list ?? []).map((file) => ({ file, path: (file as any).webkitRelativePath || "" }));
}

/** From a drop: files as they are, folders walked to the bottom. */
export async function fromDrop(transfer: DataTransfer): Promise<{ file: File; path: string }[]> {
  const entries = Array.from(transfer.items ?? [])
    .map((item) => (item.kind === "file" ? (item as any).webkitGetAsEntry?.() : null))
    .filter(Boolean);
  if (!entries.length) return Array.from(transfer.files ?? []).map((file) => ({ file, path: "" }));
  const found: { file: File; path: string }[] = [];
  for (const entry of entries) await walk(entry, found);
  return found;
}

async function walk(entry: any, found: { file: File; path: string }[]): Promise<void> {
  if (entry.isFile) {
    const file: File = await new Promise((resolve, reject) => entry.file(resolve, reject));
    // A file dropped on its own has no folder; one inside a dropped folder keeps its path ("Folder/Sub/x.mp4")
    const path = entry.fullPath.replace(/^\//, "");
    found.push({ file, path: path.includes("/") ? path : "" });
    return;
  }
  if (!entry.isDirectory) return;
  const reader = entry.createReader();
  // readEntries returns at most ~100 entries a call; it is called until it returns none
  for (;;) {
    const batch: any[] = await new Promise((resolve, reject) => reader.readEntries(resolve, reject));
    if (!batch.length) break;
    for (const child of batch) await walk(child, found);
  }
}
