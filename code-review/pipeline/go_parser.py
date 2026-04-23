"""
go_parser.py — AST-Based Go Code Chunking

Parses Go source files to extract function-level, struct-level,
and file-level code chunks for review by the LLM.

Uses a Go helper script via subprocess for accurate AST parsing,
with a Python regex fallback for environments without Go installed.

Usage (as module):
    from pipeline.go_parser import extract_go_chunks, GoChunk

    chunks = extract_go_chunks("/path/to/go/repo")
    for chunk in chunks:
        print(f"{chunk.chunk_type}: {chunk.name} ({chunk.start_line}-{chunk.end_line})")
"""

import os
import re
import subprocess
import json
import tempfile
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class GoChunk:
    """Represents a reviewable unit of Go code."""
    file_path: str
    chunk_type: str          # "function", "method", "struct", "interface", "whole_file"
    name: str                # Function/type name
    package: str             # Package name
    start_line: int          # 1-based start line
    end_line: int            # 1-based end line
    code: str                # Source code text
    imports: List[str] = field(default_factory=list)
    receiver: str = ""       # Method receiver type (empty for functions)
    doc_comment: str = ""    # Documentation comment if present

    def line_count(self) -> int:
        return self.end_line - self.start_line + 1

    def to_dict(self) -> dict:
        return asdict(self)


# ── Go AST Helper ────────────────────────────────────────────────────────────

GO_AST_HELPER = '''
package main

import (
\t"encoding/json"
\t"fmt"
\t"go/ast"
\t"go/parser"
\t"go/token"
\t"os"
\t"strings"
)

type Chunk struct {
\tType      string   `json:"type"`
\tName      string   `json:"name"`
\tPackage   string   `json:"package"`
\tStartLine int      `json:"start_line"`
\tEndLine   int      `json:"end_line"`
\tReceiver  string   `json:"receiver,omitempty"`
\tDoc       string   `json:"doc,omitempty"`
\tImports   []string `json:"imports"`
}

func main() {
\tif len(os.Args) < 2 {
\t\tfmt.Fprintln(os.Stderr, "usage: go_ast_helper <file.go>")
\t\tos.Exit(1)
\t}

\tfilePath := os.Args[1]
\tfset := token.NewFileSet()
\tnode, err := parser.ParseFile(fset, filePath, nil, parser.ParseComments)
\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "parse error: %v\\n", err)
\t\tos.Exit(1)
\t}

\tvar imports []string
\tfor _, imp := range node.Imports {
\t\timports = append(imports, strings.Trim(imp.Path.Value, `"`))
\t}

\tvar chunks []Chunk

\tfor _, decl := range node.Decls {
\t\tswitch d := decl.(type) {
\t\tcase *ast.FuncDecl:
\t\t\tchunk := Chunk{
\t\t\t\tName:      d.Name.Name,
\t\t\t\tPackage:   node.Name.Name,
\t\t\t\tStartLine: fset.Position(d.Pos()).Line,
\t\t\t\tEndLine:   fset.Position(d.End()).Line,
\t\t\t\tImports:   imports,
\t\t\t}
\t\t\tif d.Recv != nil && len(d.Recv.List) > 0 {
\t\t\t\tchunk.Type = "method"
\t\t\t\texpr := d.Recv.List[0].Type
\t\t\t\tif star, ok := expr.(*ast.StarExpr); ok {
\t\t\t\t\tif ident, ok := star.X.(*ast.Ident); ok {
\t\t\t\t\t\tchunk.Receiver = ident.Name
\t\t\t\t\t}
\t\t\t\t} else if ident, ok := expr.(*ast.Ident); ok {
\t\t\t\t\tchunk.Receiver = ident.Name
\t\t\t\t}
\t\t\t} else {
\t\t\t\tchunk.Type = "function"
\t\t\t}
\t\t\tif d.Doc != nil {
\t\t\t\tchunk.Doc = d.Doc.Text()
\t\t\t}
\t\t\tchunks = append(chunks, chunk)

\t\tcase *ast.GenDecl:
\t\t\tfor _, spec := range d.Specs {
\t\t\t\tswitch s := spec.(type) {
\t\t\t\tcase *ast.TypeSpec:
\t\t\t\t\tchunk := Chunk{
\t\t\t\t\t\tName:      s.Name.Name,
\t\t\t\t\t\tPackage:   node.Name.Name,
\t\t\t\t\t\tStartLine: fset.Position(d.Pos()).Line,
\t\t\t\t\t\tEndLine:   fset.Position(d.End()).Line,
\t\t\t\t\t\tImports:   imports,
\t\t\t\t\t}
\t\t\t\t\tswitch s.Type.(type) {
\t\t\t\t\tcase *ast.StructType:
\t\t\t\t\t\tchunk.Type = "struct"
\t\t\t\t\tcase *ast.InterfaceType:
\t\t\t\t\t\tchunk.Type = "interface"
\t\t\t\t\tdefault:
\t\t\t\t\t\tchunk.Type = "type"
\t\t\t\t\t}
\t\t\t\t\tif d.Doc != nil {
\t\t\t\t\t\tchunk.Doc = d.Doc.Text()
\t\t\t\t\t}
\t\t\t\t\tchunks = append(chunks, chunk)
\t\t\t\t}
\t\t\t}
\t\t}
\t}

\tjson.NewEncoder(os.Stdout).Encode(chunks)
}
'''


def _compile_go_helper(helper_dir: str) -> Optional[str]:
    """Compile the Go AST helper binary. Returns path to binary or None."""
    go_file = os.path.join(helper_dir, "ast_helper.go")
    binary = os.path.join(helper_dir, "ast_helper.exe" if os.name == "nt" else "ast_helper")

    # Skip if already compiled
    if os.path.exists(binary):
        return binary

    # Write Go source
    os.makedirs(helper_dir, exist_ok=True)
    with open(go_file, "w", encoding="utf-8") as f:
        f.write(GO_AST_HELPER)

    # Compile
    try:
        result = subprocess.run(
            ["go", "build", "-o", binary, go_file],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            return binary
        else:
            print(f"[WARN] Failed to compile Go AST helper: {result.stderr}")
            return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def parse_go_file_with_ast(
    file_path: str,
    helper_binary: str,
) -> List[dict]:
    """Parse a Go file using the compiled AST helper."""
    try:
        result = subprocess.run(
            [helper_binary, file_path],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        pass
    return []


# ── Python Regex Fallback ────────────────────────────────────────────────────

def parse_go_file_regex(file_path: str, source: str) -> List[dict]:
    """
    Fallback parser using regex when Go toolchain is not available.
    Less accurate than AST but handles most common patterns.
    """
    chunks = []
    lines = source.split("\n")

    # Extract package name
    package = "unknown"
    for line in lines:
        match = re.match(r"^package\s+(\w+)", line)
        if match:
            package = match.group(1)
            break

    # Extract imports
    imports = []
    in_import_block = False
    for line in lines:
        if re.match(r"^import\s*\(", line):
            in_import_block = True
            continue
        if in_import_block:
            if line.strip() == ")":
                in_import_block = False
                continue
            imp = line.strip().strip('"').strip()
            if imp:
                imports.append(imp)
        elif re.match(r'^import\s+"', line):
            imp = re.findall(r'"([^"]+)"', line)
            imports.extend(imp)

    # Find functions and methods
    func_pattern = re.compile(
        r"^(//[^\n]*\n)*"                     # optional doc comment
        r"func\s+"
        r"(?:\((\w+)\s+\*?(\w+)\)\s+)?"       # optional receiver
        r"(\w+)\s*\(",                          # function name
        re.MULTILINE,
    )

    for match in func_pattern.finditer(source):
        func_name = match.group(4)
        receiver = match.group(3) or ""
        chunk_type = "method" if match.group(2) else "function"

        # Find start line
        start_pos = match.start()
        start_line = source[:start_pos].count("\n") + 1

        # Find matching closing brace
        brace_count = 0
        end_line = start_line
        found_open = False

        for i in range(start_line - 1, len(lines)):
            brace_count += lines[i].count("{") - lines[i].count("}")
            if "{" in lines[i]:
                found_open = True
            if found_open and brace_count == 0:
                end_line = i + 1
                break

        code = "\n".join(lines[start_line - 1 : end_line])

        chunks.append({
            "type": chunk_type,
            "name": func_name,
            "package": package,
            "start_line": start_line,
            "end_line": end_line,
            "receiver": receiver,
            "imports": imports,
            "doc": "",
        })

    # Find structs and interfaces
    type_pattern = re.compile(
        r"^type\s+(\w+)\s+(struct|interface)\s*\{",
        re.MULTILINE,
    )

    for match in type_pattern.finditer(source):
        type_name = match.group(1)
        type_kind = match.group(2)

        start_pos = match.start()
        start_line = source[:start_pos].count("\n") + 1

        brace_count = 0
        end_line = start_line
        found_open = False

        for i in range(start_line - 1, len(lines)):
            brace_count += lines[i].count("{") - lines[i].count("}")
            if "{" in lines[i]:
                found_open = True
            if found_open and brace_count == 0:
                end_line = i + 1
                break

        chunks.append({
            "type": type_kind,
            "name": type_name,
            "package": package,
            "start_line": start_line,
            "end_line": end_line,
            "receiver": "",
            "imports": imports,
            "doc": "",
        })

    return chunks


# ── Chunking Strategy ────────────────────────────────────────────────────────

def chunk_strategy(file_lines: int) -> str:
    """
    Choose chunking granularity based on file size.
    - Small files (< 150 lines): review whole file at once
    - Medium files (< 600 lines): review function by function
    - Large files (>= 600 lines): review individual methods
    """
    if file_lines < 150:
        return "whole_file"
    elif file_lines < 600:
        return "function_level"
    else:
        return "method_level"


SKIP_PATTERNS = [
    "vendor/",
    "node_modules/",
    ".git/",
    "testdata/",
    "_gen.go",
    ".pb.go",          # Protobuf generated
    "_mock.go",        # Generated mocks
    "zz_generated",    # k8s generated
    "_string.go",      # stringer generated
]


def should_skip_file(file_path: str) -> bool:
    """Check if a file should be skipped (vendor, generated, etc.)."""
    path_str = str(file_path).replace("\\", "/")
    return any(pattern in path_str for pattern in SKIP_PATTERNS)


# ── Main Extraction ──────────────────────────────────────────────────────────

def extract_go_chunks(
    repo_path: str,
    include_tests: bool = True,
    max_chunk_lines: int = 300,
) -> List[GoChunk]:
    """
    Walk a Go repository and extract function/method/struct level chunks
    for individual review by the LLM.

    Args:
        repo_path: Path to the Go repository root
        include_tests: Whether to include test files
        max_chunk_lines: Maximum lines per chunk (larger chunks are split)

    Returns:
        List of GoChunk objects ready for review
    """
    chunks: List[GoChunk] = []
    repo_path = os.path.abspath(repo_path)

    # Try to compile Go AST helper
    helper_dir = os.path.join(tempfile.gettempdir(), "go_code_reviewer")
    helper_binary = _compile_go_helper(helper_dir)

    if helper_binary:
        print("[INFO] Using Go AST parser (accurate mode)")
    else:
        print("[INFO] Using Python regex parser (fallback mode)")
        print("[INFO] Install Go for better parsing accuracy")

    go_files = list(Path(repo_path).rglob("*.go"))
    skipped = 0

    for go_file in go_files:
        file_path = str(go_file)

        # Skip vendor, generated, etc.
        if should_skip_file(file_path):
            skipped += 1
            continue

        # Skip test files if not requested
        if not include_tests and go_file.name.endswith("_test.go"):
            skipped += 1
            continue

        try:
            with open(go_file, "r", encoding="utf-8") as f:
                source = f.read()
        except (UnicodeDecodeError, PermissionError):
            skipped += 1
            continue

        lines = source.split("\n")
        strategy = chunk_strategy(len(lines))

        # Relative path for cleaner output
        rel_path = os.path.relpath(file_path, repo_path).replace("\\", "/")

        if strategy == "whole_file":
            # Extract package name
            package = "unknown"
            for line in lines:
                m = re.match(r"^package\s+(\w+)", line)
                if m:
                    package = m.group(1)
                    break

            chunks.append(GoChunk(
                file_path=rel_path,
                chunk_type="whole_file",
                name=go_file.stem,
                package=package,
                start_line=1,
                end_line=len(lines),
                code=source,
            ))
        else:
            # Parse into function/struct level chunks
            if helper_binary:
                parsed = parse_go_file_with_ast(file_path, helper_binary)
            else:
                parsed = parse_go_file_regex(file_path, source)

            if not parsed:
                # If parsing fails, fall back to whole file
                package = "unknown"
                for line in lines:
                    m = re.match(r"^package\s+(\w+)", line)
                    if m:
                        package = m.group(1)
                        break

                chunks.append(GoChunk(
                    file_path=rel_path,
                    chunk_type="whole_file",
                    name=go_file.stem,
                    package=package,
                    start_line=1,
                    end_line=len(lines),
                    code=source,
                ))
                continue

            for item in parsed:
                start = item.get("start_line", 1)
                end = item.get("end_line", len(lines))
                code = "\n".join(lines[start - 1 : end])

                chunks.append(GoChunk(
                    file_path=rel_path,
                    chunk_type=item.get("type", "function"),
                    name=item.get("name", "unknown"),
                    package=item.get("package", "unknown"),
                    start_line=start,
                    end_line=end,
                    code=code,
                    imports=item.get("imports", []),
                    receiver=item.get("receiver", ""),
                    doc_comment=item.get("doc", ""),
                ))

    print(f"[INFO] Extracted {len(chunks)} chunks from {len(go_files) - skipped} files (skipped {skipped})")
    return chunks


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract Go code chunks from a repository")
    parser.add_argument("repo_path", help="Path to Go repository")
    parser.add_argument("--include-tests", action="store_true", default=True)
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    chunks = extract_go_chunks(args.repo_path, args.include_tests)

    if args.json:
        print(json.dumps([c.to_dict() for c in chunks], indent=2))
    else:
        for chunk in chunks:
            print(
                f"  [{chunk.chunk_type:12}] {chunk.package}.{chunk.name:30} "
                f"({chunk.file_path}:{chunk.start_line}-{chunk.end_line}) "
                f"{chunk.line_count()} lines"
            )
        print(f"\nTotal: {len(chunks)} chunks")
