"""Tests for PTX → pyptx Python code generator."""

import pytest
import textwrap

from pyptx.codegen import ptx_to_python
from tests.conftest import corpus_files


class TestCodegenBasic:
    def test_minimal_kernel(self):
        ptx_src = textwrap.dedent("""\
        .version 8.5
        .target sm_90a
        .address_size 64

        .visible .entry minimal()
        {
        \tret;
        }
        """)
        py = ptx_to_python(ptx_src)
        assert "from pyptx import kernel, reg, smem, ptx" in py
        assert "@kernel" in py
        assert "def minimal():" in py
        assert "ptx.ret()" in py

    def test_register_arrays(self):
        ptx_src = textwrap.dedent("""\
        .version 8.5
        .target sm_90a
        .address_size 64

        .visible .entry test()
        {
        \t.reg .b32 %r<10>;
        \t.reg .b64 %rd<5>;
        \t.reg .pred %p<3>;
        \tmov.b32 %r0, %r1;
        \tret;
        }
        """)
        py = ptx_to_python(ptx_src)
        assert 'r = reg.array(b32, 10, name="%r")' in py
        assert 'rd = reg.array(b64, 5, name="%rd")' in py
        assert 'p = reg.array(pred, 3, name="%p")' in py
        assert "r[0]" in py
        assert "r[1]" in py

    def test_special_registers(self):
        ptx_src = textwrap.dedent("""\
        .version 8.5
        .target sm_90a
        .address_size 64

        .visible .entry test()
        {
        \t.reg .b32 %r<5>;
        \tmov.u32 %r0, %tid.x;
        \tmov.u32 %r1, %ctaid.x;
        \tret;
        }
        """)
        py = ptx_to_python(ptx_src)
        assert "ptx.special.tid.x()" in py
        assert "ptx.special.ctaid.x()" in py

    def test_predicated_branch(self):
        ptx_src = textwrap.dedent("""\
        .version 8.5
        .target sm_90a
        .address_size 64

        .visible .entry test()
        {
        \t.reg .pred %p<5>;
        \t@%p0 bra DONE;
        DONE:
        \tret;
        }
        """)
        py = ptx_to_python(ptx_src)
        assert 'ptx.bra("DONE", pred=p[0])' in py
        assert 'ptx.label("DONE")' in py

    def test_negated_predicate(self):
        ptx_src = textwrap.dedent("""\
        .version 8.5
        .target sm_90a
        .address_size 64

        .visible .entry test()
        {
        \t.reg .pred %p<5>;
        \t.reg .b32 %r<5>;
        \t@!%p0 mov.b32 %r0, 0;
        \tret;
        }
        """)
        py = ptx_to_python(ptx_src)
        assert "pred=~p[0]" in py

    def test_address_operands(self):
        ptx_src = textwrap.dedent("""\
        .version 8.5
        .target sm_90a
        .address_size 64

        .visible .entry test()
        {
        \t.reg .b32 %r<5>;
        \t.reg .b64 %rd<5>;
        \tld.param.u64 %rd0, [param0];
        \tst.shared.b32 [%rd0+4], %r0;
        \tret;
        }
        """)
        py = ptx_to_python(ptx_src)
        assert 'ptx.addr("param0")' in py
        assert "ptx.addr(rd[0], 4)" in py

    def test_vector_operand(self):
        ptx_src = textwrap.dedent("""\
        .version 8.5
        .target sm_90a
        .address_size 64

        .visible .entry test()
        {
        \t.reg .b32 %r<100>;
        \t.reg .b64 %rd<50>;
        \twgmma.mma_async.sync.aligned.m64n256k16.f32.e4m3.e4m3 {%r0, %r1, %r2, %r3}, %rd0, %rd1, 1, 1, 1;
        \tret;
        }
        """)
        py = ptx_to_python(ptx_src)
        assert "[r[0], r[1], r[2], r[3]]" in py
        assert "wgmma" in py

    def test_shared_memory(self):
        ptx_src = textwrap.dedent("""\
        .version 8.5
        .target sm_90a
        .address_size 64

        .visible .entry test()
        {
        \t.shared .align 128 .b8 smem[49152];
        \tret;
        }
        """)
        py = ptx_to_python(ptx_src)
        assert 'ptx.var("shared", b8, "smem", size=49152, align=128)' in py

    def test_global_keyword_escaped(self):
        ptx_src = textwrap.dedent("""\
        .version 8.5
        .target sm_90a
        .address_size 64

        .visible .entry test()
        {
        \t.reg .b32 %r<5>;
        \t.reg .b64 %rd<5>;
        \tld.global.f32 %r0, [%rd0];
        \tret;
        }
        """)
        py = ptx_to_python(ptx_src)
        assert ".global_." in py  # global is escaped

    def test_pipe_operand(self):
        ptx_src = textwrap.dedent("""\
        .version 8.5
        .target sm_90a
        .address_size 64

        .visible .entry test()
        {
        \t.reg .pred %p<5>;
        \t.reg .b32 %r<5>;
        \tsetp.lt.f32 %p1|%p2, %r2, %r3;
        \tret;
        }
        """)
        py = ptx_to_python(ptx_src)
        assert "p[1] | p[2]" in py

    def test_type_imports(self):
        ptx_src = textwrap.dedent("""\
        .version 8.5
        .target sm_90a
        .address_size 64

        .visible .entry test()
        {
        \t.reg .b32 %r<5>;
        \t.reg .pred %p<3>;
        \t.reg .f32 %f<10>;
        \tret;
        }
        """)
        py = ptx_to_python(ptx_src)
        assert "from pyptx.types import" in py
        assert "b32" in py
        assert "pred" in py
        assert "f32" in py


class TestCodegenCorpus:
    """Run codegen on all corpus files and verify the output is valid Python."""

    @pytest.mark.parametrize("ptx_file", corpus_files(), ids=lambda p: p.name)
    def test_codegen_produces_valid_python(self, ptx_file):
        source = ptx_file.read_text()
        py = ptx_to_python(source)
        # Should be syntactically valid Python
        compile(py, f"<codegen:{ptx_file.name}>", "exec")

    @pytest.mark.parametrize("ptx_file", corpus_files(), ids=lambda p: p.name)
    def test_codegen_has_kernel_decorator(self, ptx_file):
        source = ptx_file.read_text()
        py = ptx_to_python(source)
        assert "@kernel" in py

    @pytest.mark.parametrize("ptx_file", corpus_files(), ids=lambda p: p.name)
    def test_codegen_has_imports(self, ptx_file):
        source = ptx_file.read_text()
        py = ptx_to_python(source)
        assert "from pyptx import" in py


class TestCodegenEndToEnd:
    """Generate Python code, execute it, and verify it produces PTX."""

    def test_vector_add_roundtrip(self):
        """PTX → Python → execute → PTX. Verify the Python code runs."""
        ptx_src = textwrap.dedent("""\
        .version 8.5
        .target sm_90a
        .address_size 64

        .visible .entry vadd()
        {
        \t.reg .b32 %r<5>;
        \tmov.b32 %r0, 42;
        \tadd.s32 %r1, %r0, %r0;
        \tret;
        }
        """)
        py = ptx_to_python(ptx_src)

        # Execute the generated code
        namespace: dict = {}
        exec(py, namespace)

        # The generated code should have defined a kernel
        assert "vadd" in namespace
        kernel_fn = namespace["vadd"]

        # It should be able to emit PTX
        output_ptx = kernel_fn.ptx()
        assert "mov.b32" in output_ptx
        assert "add.s32" in output_ptx
        assert "ret;" in output_ptx
