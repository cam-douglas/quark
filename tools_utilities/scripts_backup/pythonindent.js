
let imports = {};
imports['__wbindgen_placeholder__'] = module.exports;
let wasm;
const { TextEncoder, TextDecoder } = require(`util`);

let WASM_VECTOR_LEN = 0;

let cachedUint8ArrayMemory0 = null;

function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

let cachedTextEncoder = new TextEncoder('utf-8');

const encodeString = (typeof cachedTextEncoder.encodeInto === 'function'
    ? function (arg, view) {
    return cachedTextEncoder.encodeInto(arg, view);
}
    : function (arg, view) {
    const buf = cachedTextEncoder.encode(arg);
    view.set(buf);
    return {
        read: arg.length,
        written: buf.length
    };
});

function passStringToWasm0(arg, malloc, realloc) {

    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }

    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = encodeString(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function isLikeNone(x) {
    return x === undefined || x === null;
}

let cachedDataViewMemory0 = null;

function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });

cachedTextDecoder.decode();

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

function _assertClass(instance, klass) {
    if (!(instance instanceof klass)) {
        throw new Error(`expected instance of ${klass.name}`);
    }
    return instance.ptr;
}

function getArrayJsValueFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    const mem = getDataViewMemory0();
    const result = [];
    for (let i = ptr; i < ptr + 4 * len; i += 4) {
        result.push(wasm.__wbindgen_export_2.get(mem.getUint32(i, true)));
    }
    wasm.__externref_drop_slice(ptr, len);
    return result;
}

function addToExternrefTable0(obj) {
    const idx = wasm.__externref_table_alloc();
    wasm.__wbindgen_export_2.set(idx, obj);
    return idx;
}

function passArrayJsValueToWasm0(array, malloc) {
    const ptr = malloc(array.length * 4, 4) >>> 0;
    const mem = getDataViewMemory0();
    for (let i = 0; i < array.length; i++) {
        mem.setUint32(ptr + 4 * i, addToExternrefTable0(array[i]), true);
    }
    WASM_VECTOR_LEN = array.length;
    return ptr;
}
/**
 * @param {(string)[]} lines
 * @returns {IParseOutput}
 */
module.exports.parse_lines = function(lines) {
    const ptr0 = passArrayJsValueToWasm0(lines, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.parse_lines(ptr0, len0);
    return IParseOutput.__wrap(ret);
};

const IParseOutputFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_iparseoutput_free(ptr >>> 0, 1));

class IParseOutput {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(IParseOutput.prototype);
        obj.__wbg_ptr = ptr;
        IParseOutputFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        IParseOutputFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_iparseoutput_free(ptr, 0);
    }
    /**
     * @returns {boolean}
     */
    get can_hang() {
        const ret = wasm.__wbg_get_iparseoutput_can_hang(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @param {boolean} arg0
     */
    set can_hang(arg0) {
        wasm.__wbg_set_iparseoutput_can_hang(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {boolean}
     */
    get dedent_next() {
        const ret = wasm.__wbg_get_iparseoutput_dedent_next(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @param {boolean} arg0
     */
    set dedent_next(arg0) {
        wasm.__wbg_set_iparseoutput_dedent_next(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {OpenClose | undefined}
     */
    get last_closed_info() {
        const ret = wasm.__wbg_get_iparseoutput_last_closed_info(this.__wbg_ptr);
        return ret === 0 ? undefined : OpenClose.__wrap(ret);
    }
    /**
     * @param {OpenClose | undefined} [arg0]
     */
    set last_closed_info(arg0) {
        let ptr0 = 0;
        if (!isLikeNone(arg0)) {
            _assertClass(arg0, OpenClose);
            ptr0 = arg0.__destroy_into_raw();
        }
        wasm.__wbg_set_iparseoutput_last_closed_info(this.__wbg_ptr, ptr0);
    }
    /**
     * @returns {number | undefined}
     */
    get last_colon_row() {
        const ret = wasm.__wbg_get_iparseoutput_last_colon_row(this.__wbg_ptr);
        return ret[0] === 0 ? undefined : ret[1] >>> 0;
    }
    /**
     * @param {number | undefined} [arg0]
     */
    set last_colon_row(arg0) {
        wasm.__wbg_set_iparseoutput_last_colon_row(this.__wbg_ptr, !isLikeNone(arg0), isLikeNone(arg0) ? 0 : arg0);
    }
    /**
     * @returns {(RowCol)[]}
     */
    get open_bracket_stack() {
        const ret = wasm.__wbg_get_iparseoutput_open_bracket_stack(this.__wbg_ptr);
        var v1 = getArrayJsValueFromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @param {(RowCol)[]} arg0
     */
    set open_bracket_stack(arg0) {
        const ptr0 = passArrayJsValueToWasm0(arg0, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.__wbg_set_iparseoutput_open_bracket_stack(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @returns {LastSeenIndenters}
     */
    get last_seen_indenters() {
        const ret = wasm.__wbg_get_iparseoutput_last_seen_indenters(this.__wbg_ptr);
        return LastSeenIndenters.__wrap(ret);
    }
    /**
     * @param {LastSeenIndenters} arg0
     */
    set last_seen_indenters(arg0) {
        _assertClass(arg0, LastSeenIndenters);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_iparseoutput_last_seen_indenters(this.__wbg_ptr, ptr0);
    }
}
module.exports.IParseOutput = IParseOutput;

const LastSeenIndentersFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_lastseenindenters_free(ptr >>> 0, 1));

class LastSeenIndenters {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(LastSeenIndenters.prototype);
        obj.__wbg_ptr = ptr;
        LastSeenIndentersFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        LastSeenIndentersFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_lastseenindenters_free(ptr, 0);
    }
    /**
     * @returns {number | undefined}
     */
    get if_() {
        const ret = wasm.__wbg_get_lastseenindenters_if_(this.__wbg_ptr);
        return ret[0] === 0 ? undefined : ret[1] >>> 0;
    }
    /**
     * @param {number | undefined} [arg0]
     */
    set if_(arg0) {
        wasm.__wbg_set_lastseenindenters_if_(this.__wbg_ptr, !isLikeNone(arg0), isLikeNone(arg0) ? 0 : arg0);
    }
    /**
     * @returns {number | undefined}
     */
    get for_() {
        const ret = wasm.__wbg_get_lastseenindenters_for_(this.__wbg_ptr);
        return ret[0] === 0 ? undefined : ret[1] >>> 0;
    }
    /**
     * @param {number | undefined} [arg0]
     */
    set for_(arg0) {
        wasm.__wbg_set_lastseenindenters_for_(this.__wbg_ptr, !isLikeNone(arg0), isLikeNone(arg0) ? 0 : arg0);
    }
    /**
     * @returns {number | undefined}
     */
    get try_() {
        const ret = wasm.__wbg_get_lastseenindenters_try_(this.__wbg_ptr);
        return ret[0] === 0 ? undefined : ret[1] >>> 0;
    }
    /**
     * @param {number | undefined} [arg0]
     */
    set try_(arg0) {
        wasm.__wbg_set_lastseenindenters_try_(this.__wbg_ptr, !isLikeNone(arg0), isLikeNone(arg0) ? 0 : arg0);
    }
    /**
     * @returns {number | undefined}
     */
    get while_() {
        const ret = wasm.__wbg_get_lastseenindenters_while_(this.__wbg_ptr);
        return ret[0] === 0 ? undefined : ret[1] >>> 0;
    }
    /**
     * @param {number | undefined} [arg0]
     */
    set while_(arg0) {
        wasm.__wbg_set_lastseenindenters_while_(this.__wbg_ptr, !isLikeNone(arg0), isLikeNone(arg0) ? 0 : arg0);
    }
}
module.exports.LastSeenIndenters = LastSeenIndenters;

const OpenCloseFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_openclose_free(ptr >>> 0, 1));

class OpenClose {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(OpenClose.prototype);
        obj.__wbg_ptr = ptr;
        OpenCloseFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        OpenCloseFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_openclose_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get open() {
        const ret = wasm.__wbg_get_openclose_open(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} arg0
     */
    set open(arg0) {
        wasm.__wbg_set_openclose_open(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get close() {
        const ret = wasm.__wbg_get_openclose_close(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} arg0
     */
    set close(arg0) {
        wasm.__wbg_set_openclose_close(this.__wbg_ptr, arg0);
    }
}
module.exports.OpenClose = OpenClose;

const RowColFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_rowcol_free(ptr >>> 0, 1));

class RowCol {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(RowCol.prototype);
        obj.__wbg_ptr = ptr;
        RowColFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    static __unwrap(jsValue) {
        if (!(jsValue instanceof RowCol)) {
            return 0;
        }
        return jsValue.__destroy_into_raw();
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        RowColFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_rowcol_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get row() {
        const ret = wasm.__wbg_get_openclose_open(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} arg0
     */
    set row(arg0) {
        wasm.__wbg_set_openclose_open(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get col() {
        const ret = wasm.__wbg_get_openclose_close(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} arg0
     */
    set col(arg0) {
        wasm.__wbg_set_openclose_close(this.__wbg_ptr, arg0);
    }
}
module.exports.RowCol = RowCol;

module.exports.__wbg_rowcol_unwrap = function(arg0) {
    const ret = RowCol.__unwrap(arg0);
    return ret;
};

module.exports.__wbg_rowcol_new = function(arg0) {
    const ret = RowCol.__wrap(arg0);
    return ret;
};

module.exports.__wbindgen_string_get = function(arg0, arg1) {
    const obj = arg1;
    const ret = typeof(obj) === 'string' ? obj : undefined;
    var ptr1 = isLikeNone(ret) ? 0 : passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    var len1 = WASM_VECTOR_LEN;
    getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
    getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
};

module.exports.__wbindgen_throw = function(arg0, arg1) {
    throw new Error(getStringFromWasm0(arg0, arg1));
};

module.exports.__wbindgen_init_externref_table = function() {
    const table = wasm.__wbindgen_export_2;
    const offset = table.grow(4);
    table.set(0, undefined);
    table.set(offset + 0, undefined);
    table.set(offset + 1, null);
    table.set(offset + 2, true);
    table.set(offset + 3, false);
    ;
};

const path = require('path').join(__dirname, 'pythonindent_bg.wasm');
const bytes = require('fs').readFileSync(path);

const wasmModule = new WebAssembly.Module(bytes);
const wasmInstance = new WebAssembly.Instance(wasmModule, imports);
wasm = wasmInstance.exports;
module.exports.__wasm = wasm;

wasm.__wbindgen_start();

