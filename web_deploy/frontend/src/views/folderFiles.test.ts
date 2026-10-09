import { accepts, dedupe, pick, uploadName } from "./folderFiles";

const file = (name: string) => new File(["x"], name);

describe("folder files", () => {
  it("names a file by its path inside the picked folder", () => {
    expect(uploadName("Lectures/Unit 3/Part I.mp4")).toBe("Unit 3 - Part I.mp4");
    expect(uploadName("Lectures/Intro.mp4")).toBe("Intro.mp4");
    expect(uploadName("Intro.mp4")).toBe("Intro.mp4");
  });

  it("numbers a repeated name instead of overwriting the first", () => {
    expect(dedupe(["a.mp4", "b.mp4", "A.mp4", "a.mp4", "notes"])).toEqual(["a.mp4", "b.mp4", "A (2).mp4", "a (3).mp4", "notes"]);
  });

  it("keeps only the chosen kind and says how many were left out", () => {
    const picked = pick([
      { file: file("Part I.mp4"), path: "Lectures/Unit 3/Part I.mp4" },
      { file: file("Part I.MP4"), path: "Lectures/Unit 4/Part I.MP4" },
      { file: file("notes.pdf"), path: "Lectures/notes.pdf" },
      { file: file(".DS_Store"), path: "Lectures/.DS_Store" },
    ], ".mp4,.mkv");

    expect(picked.files.map((f) => f.name)).toEqual(["Unit 3 - Part I.mp4", "Unit 4 - Part I.MP4"]);
    expect(picked.skipped).toBe(2);
    expect(picked.folder).toBe("Lectures");
  });

  it("files picked one by one keep their own names and have no folder", () => {
    const picked = pick([{ file: file("a.pdf"), path: "" }, { file: file("a.pdf"), path: "" }], ".pdf");
    expect(picked.files.map((f) => f.name)).toEqual(["a.pdf", "a (2).pdf"]);
    expect(picked.folder).toBeNull();
  });

  it("matches extensions without regard to case", () => {
    expect(accepts("Talk.MKV", ".mp4,.mkv")).toBe(true);
    expect(accepts("talk.mp3", ".mp4,.mkv")).toBe(false);
  });
});
