import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const workbookPaths = process.argv.slice(2);

if (workbookPaths.length === 0) {
  throw new Error("Pass at least one .xlsx path.");
}

for (const workbookPath of workbookPaths) {
  const input = await FileBlob.load(workbookPath);
  const workbook = await SpreadsheetFile.importXlsx(input);

  console.log(JSON.stringify({ type: "workbook", path: workbookPath }));
  const sheets = await workbook.inspect({
    kind: "sheet",
    include: "id,name",
    maxChars: 20000,
  });
  console.log(sheets.ndjson);

  for (let index = 0; index < 100; index += 1) {
    let sheet;
    try {
      sheet = workbook.worksheets.getItemAt(index);
    } catch {
      break;
    }
    if (!sheet) {
      break;
    }

    const usedRange = sheet.getUsedRange(true);
    console.log(
      JSON.stringify({
        type: "used_range",
        index,
        name: sheet.name,
        address: usedRange?.address ?? null,
      }),
    );

    const sample = await workbook.inspect({
      kind: "region",
      sheetId: sheet.name,
      range: "A1:Z20",
      maxChars: 12000,
      tableMaxRows: 20,
      tableMaxCols: 26,
      tableMaxCellChars: 120,
    });
    console.log(sample.ndjson);
  }
}
